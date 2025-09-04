import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig)
from vllm.model_executor.models.qwen2_5_omni_thinker import (Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniAudioEncoder,Qwen2_5_VisionTransformer)
from typing import Iterable
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix, AutoWeightsLoader

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
