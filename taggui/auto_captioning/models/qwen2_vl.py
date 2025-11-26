"""
Qwen2-VL and Qwen2.5-VL model implementation.

These models require specific chat template formatting for image inputs.
"""

import torch
from PIL import Image as PilImage
from transformers import AutoProcessor, AutoModelForVision2Seq

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image


class Qwen2VL(AutoCaptioningModel):
    """
    Qwen2-VL and Qwen2.5-VL model with proper chat template formatting.

    These models expect inputs in a specific chat format with image placeholders.
    """

    # Use AutoModelForVision2Seq to automatically select the right class
    transformers_model_class = AutoModelForVision2Seq

    def get_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

    @staticmethod
    def get_default_prompt() -> str:
        return "Describe this image in detail."

    def get_model_inputs(self, image_prompt: str, image: Image):
        """
        Format inputs using Qwen2.5-VL chat template.

        Qwen2.5-VL requires a specific message format with image content.
        """
        pil_image = self.load_image(image)

        # Build the messages in Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": image_prompt or self.get_default_prompt()}
                ]
            }
        ]

        # Apply chat template to get the formatted text
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process with the formatted text and image
        model_inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.device, **self.dtype_argument)

        return model_inputs

    def get_caption_from_generated_tokens(
            self, generated_token_ids: torch.Tensor, image_prompt: str) -> str:
        """
        Decode generated tokens, removing the input prompt.
        """
        # Decode the full output
        generated_text = self.processor.batch_decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # The output typically includes the assistant's response after the prompt
        # Try to extract just the response part
        if "assistant" in generated_text.lower():
            # Split on common assistant markers
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                generated_text = parts[-1].strip()
                # Remove leading colon or newline
                generated_text = generated_text.lstrip(":").strip()

        # Clean up any remaining template artifacts
        generated_text = generated_text.strip()

        # Apply caption start if specified
        if self.caption_start.strip():
            caption = f'{self.caption_start.strip()} {generated_text}'
        else:
            caption = generated_text

        caption = caption.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')

        return caption
