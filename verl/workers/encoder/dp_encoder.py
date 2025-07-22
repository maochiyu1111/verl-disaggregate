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
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
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


    def _forward_micro_batch(self, micro_batch) -> Tuple[torch.Tensor]:
        """
        Returns:
            encoder_embed
        """

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
        else:
            raise NotImplementedError

        video_embed, image_embed = self.encoder_module.encoder_forward(
            **multi_modal_inputs,
        )  # prevent model thinks we are generating

    

        return video_embed, image_embed
    
    # qzy note：先不处理encoder需要更新的情况
    # def _optimizer_step(self):
    #     assert self.config.grad_clip is not None

    #     if isinstance(self.Encoder_module, FSDP):
    #         grad_norm = self.Encoder_module.clip_grad_norm_(max_norm=self.config.grad_clip)
    #     elif isinstance(self.Encoder_module, FSDPModule):
    #         grad_norm = fsdp2_clip_grad_norm_(self.Encoder_module.parameters(), max_norm=self.config.grad_clip)
    #     else:
    #         grad_norm = torch.nn.utils.clip_grad_norm_(self.Encoder_module.parameters(), max_norm=self.config.grad_clip)

    #     # if grad_norm is not finite, skip the update
    #     if not torch.isfinite(grad_norm):
    #         print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
    #         self.Encoder_optimizer.zero_grad()
    #     else:
    #         self.Encoder_optimizer.step()
    #     return grad_norm

    @GPUMemoryLogger(role="dp Encoder", logger=logger)
    def extract_feature(self, data: DataProto) -> torch.Tensor:
        """extract features given a batch of data
        """
        # set to eval
        self.encoder_module.eval()

        # qzy note：目前先保持与actor相同的mbs，后期应做到分离，encoder部分不使用use_dynamic_bsz
        micro_batch_size = data.meta_info["micro_batch_size"]

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(non_tensor_select_keys).chunk(num_micro_batches)
        else:
            raise NotImplementedError

        image_embeds_list = []
        video_embeds_list = []

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                image_embeds, video_embeds = self._forward_micro_batch(micro_batch)
            image_embeds_list.append(image_embeds)
            video_embeds_list.append(video_embeds)

        image_embeds = torch.concat(image_embeds_list, dim=0)
        video_embeds = torch.concat(video_embeds_list, dim=0)

        return image_embeds, video_embeds

  