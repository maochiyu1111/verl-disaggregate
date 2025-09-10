import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig)
from vllm.model_executor.models.qwen2_5_omni_thinker import (Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniAudioEncoder,Qwen2_5_VisionTransformer,)
from typing import Iterable
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix, AutoWeightsLoader
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioInputs, Qwen2AudioProcessingInfo,
    _get_feat_extract_output_lengths)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer, Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs, Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLProcessingInfo, Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs, Qwen2_5_VLVideoPixelInputs)
def init_without_encoder_patch(self, *, vllm_config, prefix: str = ""):
    super(Qwen2_5OmniThinkerForConditionalGeneration, self).__init__()
    thinker_config: Qwen2_5OmniThinkerConfig = (
        vllm_config.model_config.hf_config.thinker_config)
    quant_config = vllm_config.quant_config
    multimodal_config = vllm_config.model_config.multimodal_config
    self.config = thinker_config
    self.multimodal_config = multimodal_config

    # force "use_flash_attention_2=True" to audio tower to align
    # the results.
    # if flash_attn is not None:
    #     audio_config = thinker_config.audio_config
    #     audio_config._attn_implementation_autoset = True
    #     audio_config._attn_implementation = "flash_attention_2"
    # else:
    #     logger.warning(
    #         "flash_attn is not available, the model may not yield the "
    #         "exactly same result as the transformers implementation "
    #         "in the audio tower part.")

    # self.audio_tower = Qwen2_5OmniAudioEncoder(thinker_config.audio_config)
    # self.visual = Qwen2_5_VisionTransformer(
    #     vision_config=thinker_config.vision_config,
    #     norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
    #     quant_config=quant_config,
    #     prefix=maybe_prefix(prefix, "visual"),
    # )
    self.dtype = torch.bfloat16
    self.spatial_merge_size = 2
    self.quant_config = quant_config
    self.language_model = init_vllm_registered_model(
        vllm_config=vllm_config,
        prefix=maybe_prefix(prefix, "language_model"),
        hf_config=thinker_config.text_config,
        architectures=["Qwen2ForCausalLM"],
    )

    self.make_empty_intermediate_tensors = (
        self.language_model.make_empty_intermediate_tensors)

def load_weights(self, weights: Iterable[tuple[str,
                                                torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=["talker.", "token2wav.","thinker.model"],
    )
    loaded_weights = loader.load_weights(weights,
                                            mapper=self.hf_to_vllm_mapper)

    return loaded_weights

from typing import Union, Mapping
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.inputs import MultiModalKwargs
def _apply_hf_processor_main_patch(
    self,
    prompt: Union[str, list[int]],
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    *,
    enable_hf_prompt_update: bool,
) -> tuple[list[int], MultiModalKwargs, bool]:
    """
    Qwen2.5-Omni reimplements this function to handle text only.
    """
    # if isinstance(prompt, str):
    #     if enable_hf_prompt_update:
    #         return self._apply_hf_processor_text_mm(
    #             prompt_text=prompt,
    #             mm_items=mm_items,
    #             hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #         )
    #     tokenizer = self.info.get_tokenizer()
    #     prompt_ids = encode_tokens(tokenizer, prompt)
    # else:
    prompt_ids = self._apply_hf_processor_tokens_only(prompt)
    # prompt_ids = prompt["raw_prompt_ids"]
    mm_kwargs = self._apply_hf_processor_mm_only(
        mm_items=mm_items,
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )
    # mm_kwargs = MultiModalKwargs.from_items(mm_items)
    # mm_kwargs = MultiModalKwargs.as_kwargs
    return prompt_ids, mm_kwargs, False


def process_audio_input(
    self,
    audio_input,
    audio_hashes: list[str] = None,
    cached_audio_features: torch.Tensor = None,
) -> torch.Tensor:

    # input_features = audio_input["input_features"]
    # audio_feature_lengths = audio_input["audio_feature_lengths"]
    # if input_features.ndim == 3:
    #     assert input_features.shape[0] == 1
    #     input_features = input_features.squeeze(0)
    # if audio_feature_lengths.ndim == 2:
    #     assert audio_feature_lengths.shape[
    #         0] == 1 or audio_feature_lengths.shape[1] == 1
    #     if audio_feature_lengths.shape[0] == 1:
    #         audio_feature_lengths = audio_feature_lengths.squeeze(0)
    #     else:
    #         audio_feature_lengths = audio_feature_lengths.squeeze(1)

    # audio_feat_lengths, audio_output_lengths = (
    #     self.audio_tower._get_feat_extract_output_lengths(
    #         audio_feature_lengths))

    # audio_outputs = self.audio_tower(
    #     input_features.to(self.audio_tower.dtype),
    #     feature_lens=audio_feature_lengths,
    #     aftercnn_lens=audio_feat_lengths,
    # )
    # #already did split 
    audio_features = audio_input
    return audio_features

def process_image_input(
        self,
        image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
    if image_input["type"] == "image_embeds":
        return image_input["image_embeds"].type(self.dtype)
    #fix dtype
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    pixel_values = image_input["pixel_values"].type(self.dtype)
    image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
    # Split concatenated embeddings for each image item.
    merge_size = self.spatial_merge_size
    sizes = grid_thw.prod(-1) // merge_size // merge_size

    return image_embeds.split(sizes.tolist())


def apply_hf_processor_mm_only(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> MultiModalKwargs:
    """
    Qwen2.5-Omni reimplements this function to handle `use_audio_in_video`.
    """
    mm_counts = mm_items.get_all_counts()

    use_audio_in_video = hf_processor_mm_kwargs.get(
        "use_audio_in_video", False)
    if use_audio_in_video and "video" in mm_counts:
        assert "audio" in mm_counts
        mm_counts["audio"] -= mm_counts["video"]
    processed_data, passthrough_data = self._get_hf_mm_data(mm_items)
    processed_data.update(passthrough_data)
    mm_kwargs = MultiModalKwargs.from_hf_inputs(
        processed_data,
        self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
    )
    # _, mm_kwargs, _ = self._apply_hf_processor_text_mm(
    #     prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
    #     mm_items=mm_items,
    #     hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    # )

    return mm_kwargs