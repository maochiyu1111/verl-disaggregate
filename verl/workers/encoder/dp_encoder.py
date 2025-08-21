# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Encoder
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.device import get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.encoder import BasePPOEncoder

__all__ = ["DataParallelPPOEncoder"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# qzy note：从actor修改而来
class DataParallelPPOEncoder(BasePPOEncoder):
    def __init__(self, config, encoder_module: nn.Module, encoder_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.encoder_module = encoder_module
        self.encoder_optimizer = encoder_optimizer

        # qzy note：不清楚encoder中是否存在padding，fused kernel，先保留
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Encoder use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Encoder use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.device_name = get_device_name()


    def _forward_micro_batch(self, micro_batch) -> Tuple[torch.Tensor]:
        """
        Returns:
            encoder_embed
        """
        device = torch.device(self.device_name)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0).to(device)
        else:
            raise NotImplementedError

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # from verl.models.transformers.qwen2_5_vl import encoder_forward
            # image_embed, video_embed = encoder_forward
            image_embed, video_embed = self.encoder_module.encoder_forward(
                **multi_modal_inputs,
            )  # prevent model thinks we are generating

        return  image_embed , video_embed
    
    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.encoder_module, FSDP):
            grad_norm = self.encoder_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.encoder_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.encoder_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.encoder_optimizer.zero_grad()
        else:
            self.encoder_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp Encoder", logger=logger)
    def extract_feature(self, data: DataProto) -> torch.Tensor:
        """extract features given a batch of data
        """
        # set to eval
        self.encoder_module.eval()

        # qzy note：目前先保持与actor相同的mbs，后期应做到分离，encoder部分不使用use_dynamic_bsz
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if not has_multi_modal_inputs:
            raise NotImplementedError
        
        non_tensor_select_keys = ["multi_modal_inputs"]
        data = data.select(non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)        

        image_embeds_list = []
        video_embeds_list = []

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                image_embeds, video_embeds = self._forward_micro_batch(model_inputs)
            image_embeds_list.append(image_embeds)
            video_embeds_list.append(video_embeds)

        def flatten_embeds(embeds_list: list):
            # 可能其中一个是None列表
            if not embeds_list or embeds_list[0] is None:
                return None
            # DataProto.from_dict会将non_tensor转成numpy数组，不支持bf16
            return [tensor.cpu().to(dtype=torch.float32) for tensor_tuple in embeds_list for tensor in tensor_tuple]
        
            
        image_embeds = flatten_embeds(image_embeds_list)
        video_embeds = flatten_embeds(video_embeds_list)
            
        return image_embeds, video_embeds
    
    @GPUMemoryLogger(role="dp Encoder", logger=logger)
    def extract_feature_train(self, data: DataProto) -> torch.Tensor:
        """extract features given a batch of data
        """
        # set to train
        self.encoder_module.train()

        # qzy note：目前先保持与actor相同的mbs，后期应做到分离，encoder部分不使用use_dynamic_bsz
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if not has_multi_modal_inputs:
            raise NotImplementedError
        
        non_tensor_select_keys = ["multi_modal_inputs"]
        data = data.select(non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)     
        self.image_embeds_list = []
        self.video_embeds_list = []
        for batch_idx, mini_batch in enumerate(mini_batches):
            self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                with torch.enable_grad():
                    image_embeds, video_embeds = self._forward_micro_batch(model_inputs)
                self.image_embeds_list.append(image_embeds)
                self.video_embeds_list.append(video_embeds)

        def flatten_embeds(embeds_list: list):
            # 可能其中一个是None列表
            if not embeds_list or embeds_list[0] is None:
                return None
            return [tensor.detach().cpu().to(dtype=torch.float32) for tensor in embeds_list]
        
            
        image_embeds = flatten_embeds(self.image_embeds_list)
        video_embeds = flatten_embeds(self.video_embeds_list)
        return image_embeds, video_embeds

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto, encoder_input: DataProto):
        # make sure we are in training mode
        # 想方法拿到每个microbatch的gradient
        self.encoder_module.train()
        # mini_batches only have gradients
        mini_batches = data.split(self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)   
        mini_batches_input =  encoder_input.split(self.config.ppo_mini_batch_size)
        # metrics = {}
        for _ in range(self.config.ppo_epochs):
            # global_index = 0
            for mini_batch, mini_batch_input in zip(mini_batches, mini_batches_input):
                
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(1)
                micro_batches_input = mini_batch_input.split(self.config.ppo_micro_batch_size_per_gpu)
                self.encoder_optimizer.zero_grad()

                success_count = 0
                for micro_batch, micro_batch_input in zip(micro_batches, micro_batches_input):
                    # current_image_embed = self.image_embeds_list[global_index]
                    # current_video_embed = self.video_embeds_list[global_index]
                    with torch.enable_grad():
                        current_image_embed, current_video_embed = self._forward_micro_batch({**micro_batch_input.non_tensor_batch})
                    image_grad = micro_batch.non_tensor_batch["image_embed_grad"][0].to(dtype=torch.bfloat16, device=torch.cuda.current_device()) if "image_embed_grad" in micro_batch.non_tensor_batch.keys() else None
                    video_grad = micro_batch.non_tensor_batch["video_embed_grad"][0].to(dtype=torch.bfloat16, device=torch.cuda.current_device()) if "video_embed_grad" in micro_batch.non_tensor_batch.keys() else None
                    if current_image_embed is not None and image_grad is not None:
                        assert current_image_embed.shape == image_grad.shape
                        current_image_embed.backward(gradient=image_grad, retain_graph=True)

                    if current_video_embed is not None and video_grad is not None:
                        assert current_video_embed.shape == video_grad.shape
                        current_video_embed.backward(gradient=video_grad)

                breakpoint()
                grad_norm = self._optimizer_step()
                # 先不传入这部分的gradient
                # mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                # append_to_dict(metrics, mini_batch_metrics)
        self.image_embeds_list.clear()
        self.video_embeds_list.clear()
        self.encoder_optimizer.zero_grad()
        # return metrics