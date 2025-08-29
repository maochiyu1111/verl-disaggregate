import gc

def apply_monkey_patch_encoder():
    from verl.models.transformers.qwen2_5_vl import encoder_forward
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    Qwen2_5_VisionTransformerPretrainedModel.encoder_forward = encoder_forward

def vllm_monkey_patch_llm(disaggregate=True):
    from verl.models.vllm.qwen2_5_vl import process_image_input, init_without_encoder, load_weights_without_encoder
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    Qwen2_5_VLForConditionalGeneration.__init__ = init_without_encoder
    Qwen2_5_VLForConditionalGeneration._process_image_input = process_image_input
    # Qwen2_5_VLForConditionalGeneration.load_weights = load_weights_without_encoder

    from vllm.distributed.parallel_state import get_pp_group
    from vllm.utils import cdiv
    from vllm.multimodal.inputs import MultiModalKwargs
    from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs

    def custom_profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if not disaggregate and (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):

            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            max_tokens_by_modality_dict = self.mm_registry \
                .get_max_tokens_per_item_by_nonzero_modality(self.model_config)
            dummy_data_modality, max_tokens_per_mm_item = max(
                max_tokens_by_modality_dict.items(), key=lambda item: item[1])

            # Check how many items of this modality can be supported by
            # the encoder budget.
            encoder_budget = min(self.max_num_encoder_input_tokens,
                                 self.encoder_cache_size)

            max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                                   max_tokens_per_mm_item)

            # Check how many items of this modality can be supported by
            # the decoder budget.
            max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(
                self.model_config)[dummy_data_modality]

            # NOTE: We do not consider max_num_batched_tokens on purpose
            # because the multimodal embeddings can be generated in advance
            # and chunked prefilled.
            max_num_mm_items_decoder_budget = self.max_num_reqs * \
                max_mm_items_per_req

            max_num_mm_items = min(max_num_mm_items_encoder_budget,
                                   max_num_mm_items_decoder_budget)

            # logger.info(
            #     "Encoder cache will be initialized with a budget of %s tokens,"
            #     " and profiled with %s %s items of the maximum feature size.",
            #     encoder_budget, max_num_mm_items, dummy_data_modality)

            # Create dummy batch of multimodal inputs.
            dummy_mm_kwargs = self.mm_registry.get_decoder_dummy_data(
                model_config=self.model_config,
                seq_len=self.max_num_tokens,
                mm_counts={
                    dummy_data_modality: 1
                },
            ).multi_modal_data

            batched_dummy_mm_inputs = MultiModalKwargs.batch(
                [dummy_mm_kwargs] * max_num_mm_items,
                pin_memory=self.pin_memory)
            batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_dummy_mm_inputs,
                device=self.device,
            )

            # Run multimodal encoder.
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs)

            sanity_check_mm_encoder_outputs(
                dummy_encoder_outputs,
                expected_num_items=max_num_mm_items,
            )

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        hidden_states = self._dummy_run(self.max_num_tokens)
        if get_pp_group().is_last_rank:
            sampler_output = self._dummy_sampler_run(hidden_states)
        else:
            sampler_output = None
        self._sync_device()
        del hidden_states, sampler_output
        self.encoder_cache.clear()
        gc.collect()

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    GPUModelRunner.profile_run = custom_profile_run
