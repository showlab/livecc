# modified from transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_inputs_for_generation
def prepare_multiturn_multimodal_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    **kwargs,
):
    # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
    
    model_inputs = super(self.__class__, self).prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        position_ids=position_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        use_cache=use_cache,
        **kwargs,
    )

    # Qwen2-VL position_ids are prepareed with rope_deltas in forward
    model_inputs["position_ids"] = None

    if model_inputs["cache_position"][0] != 0 and (model_inputs['input_ids'] != self.config.video_token_id).all(): # NOTE: here we consider streaming
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None

    return model_inputs
