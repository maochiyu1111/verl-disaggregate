# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass
from importlib.metadata import version
from typing import Optional, Union

import torch
import torch.nn as nn
from packaging.version import Version
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)

from transformers.utils import is_torchdynamo_compiling


@dataclass
class Qwen2_5_VLCausalLMOutputForPPO(Qwen2_5_VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_base_model_old_api(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
    r"""
    Copy paste Qwen2_5_VL's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/qwen2_5_vl.py
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                    f"features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, "
                    f"features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    return outputs


def forward_base_model_new_api(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
    r"""
    Copy paste Qwen2_5_VL's forward
    https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1384
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    return outputs


def forward_with_torch_backend(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> tuple | Qwen2_5_VLCausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if Version(version("transformers")) < Version("4.52.0"):
        forward_base_model = forward_base_model_old_api
    else:
        forward_base_model = forward_base_model_new_api
    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        second_per_grid_ts=second_per_grid_ts,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return Qwen2_5_VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )


def forward_with_triton_backend(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> tuple | Qwen2_5_VLCausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    if Version(version("transformers")) < Version("4.52.0"):
        forward_base_model = forward_base_model_old_api
    else:
        forward_base_model = forward_base_model_new_api
    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        second_per_grid_ts=second_per_grid_ts,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return Qwen2_5_VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )
    
    
# for disaggregate
# forward for LLM
# patch Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
class CustomQwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_embed: torch.Tensor = None,
        video_embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            # breakpoint()

        if pixel_values is not None:
            image_embeds = image_embed

            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (image_mask).sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = video_embed

            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            n_video_tokens = (video_mask).sum()
            n_video_features = video_embeds.shape[0]
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
    

# forward for encoder


def encoder_forward(
    self,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    image_embeds = None
    video_embeds = None
    
    if pixel_values is not None:
        pixel_values = pixel_values.type(self.dtype)
        image_embeds = self(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(self.dtype)
        video_embeds = self(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        
    return image_embeds, video_embeds