from typing import Tuple

from transformers import AutoProcessor, LlavaForConditionalGeneration, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-interleave")
class LLaVAInterleaveModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        processor.patch_size = 14                     # ViT-L/14
        processor.vision_feature_select_strategy = "patch"
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config