import torch
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLImageInputs
#from verl.utils.logger import dist_log
from transformers import AutoConfig

def process_image_input(
        self,
        image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2
    grid_thw_list = grid_thw.tolist()

    # dist_log(f'Custom process_image_input', ranks=[0])

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.dtype)
        #dist_log(f'process_image_input pixel_values: {pixel_values.shape} {pixel_values.sum()} {grid_thw.shape}')
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
        #dist_log(f'process_image_input image_embeds: {image_embeds.shape} {image_embeds.sum()}')

    #rank = torch.distributed.get_rank()
    #torch.save(image_input, f"/workspace/yym/RLHF/verl-disaggregate/verl/models/vllm/image_inputs{rank}.pt")
    #torch.save(image_embeds, f"/workspace/yym/RLHF/verl-disaggregate/verl/models/vllm/image_embeds{rank}.pt")
    # if torch.distributed.get_rank() == 0:
    #     if pixel_values.shape[0] == 22620:
    #        torch.save(image_input, "/workspace/yym/RLHF/verl-disaggregate/verl/models/vllm/image_inputs.pt")
    #        torch.save(image_embeds, "/workspace/yym/RLHF/verl-disaggregate/verl/models/vllm/image_embeds.pt")

    # Split concatenated embeddings for each image item.
    merge_size = self.spatial_merge_size
    sizes = grid_thw.prod(-1) // merge_size // merge_size

    return image_embeds.split(sizes.tolist())


from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformer
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix, AutoWeightsLoader
from typing import Iterable

def init_without_encoder(self, *, vllm_config, prefix: str = ""):
    super(Qwen2_5_VLForConditionalGeneration, self).__init__()
    
    config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    multimodal_config = vllm_config.model_config.multimodal_config

    self.config = config
    self.multimodal_config = multimodal_config

    self.dtype = torch.bfloat16
    self.spatial_merge_size = 2

    self.visual = None
    # self.visual = Qwen2_5_VisionTransformer(
    #     config.vision_config,
    #     norm_eps=getattr(config, "rms_norm_eps", 1e-6),
    #     quant_config=self._maybe_ignore_quant_config(quant_config),
    #     prefix=maybe_prefix(prefix, "visual"),
    # )
    # print(f'Init Qwen2_5_VL without Encoder: {self.visual.dtype}')

    self.language_model = init_vllm_registered_model(
        vllm_config=vllm_config,
        prefix=maybe_prefix(prefix, "language_model"),
        architectures=["Qwen2ForCausalLM"],
    )

    self.make_empty_intermediate_tensors = (
        self.language_model.make_empty_intermediate_tensors)
    
def load_weights_without_encoder(self, weights: Iterable[tuple[str,torch.Tensor]]) -> set[str]:
    skip_prefixes = []
    if self.visual is None:
        skip_prefixes.extend(["visual."])
    loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)