"""
Caption enhancement utilities using tag coverage analysis and LLM fusion.

This module provides functionality to analyse VLM-generated captions against
WD Tagger tags and optionally enhance them using an LLM for better detail.
"""

import logging
import re
from typing import List, Optional, Tuple

import torch

from utils.settings import get_settings

logger = logging.getLogger(__name__)

# Default enhancement prompts (used if not customised in settings)
DEFAULT_LIGHT_ENHANCEMENT_PROMPT = """Rewrite this image caption to add the missing details.

Original: {original_desc}
Add these details: {missing_tags}

Rules:
- Maximum 25 words
- Only describe what is visible
- No metaphors or flowery language
- Be factual and direct
- Do not include any text found in the image
- Do not include any reference to the prompt in the output or mention attributes

Rewritten caption:"""

DEFAULT_HEAVY_REWRITE_PROMPT = """Write a brief image caption using these detected attributes.

Context: {original_desc}
Attributes to include: {all_tags}

Rules:
- Maximum 30 words
- Only describe what is visible
- No metaphors, poetry, or flowery language
- Be factual and direct
- One or two sentences only
- Do not include any text found in the image
- Do not include any reference to the prompt or attributes in the caption, stick to what's visual

Caption:"""


class CaptionEnhancer:
    """
    Enhances VLM captions by analyzing tag coverage and using LLM fusion.

    Supports two modes:
    - Analysis only: Calculate coverage scores without enhancement
    - Full enhancement: Use LLM to incorporate missing tags naturally
    """

    # Tag importance categories
    HIGH_PRIORITY_TAGS = {
        # Physical attributes
        'hair', 'eye', 'eyes', 'skin', 'face',
        # Clothing details
        'dress', 'shirt', 'skirt', 'pants', 'suit', 'coat', 'jacket',
        'shoes', 'boots', 'heels', 'hat', 'cap', 'glasses',
        # Colours (when combined with above)
        'black', 'white', 'red', 'blue', 'green', 'yellow',
        'pink', 'purple', 'brown', 'grey', 'gray', 'orange',
        # Specific descriptors
        'long', 'short', 'curly', 'straight', 'blonde', 'brunette',
    }

    LOW_PRIORITY_TAGS = {
        # Generic tags
        '1girl', '1boy', '2girls', '2boys', '3girls', '3boys',
        'solo', 'multiple', 'person', 'people',
        # Viewpoint tags
        'looking at viewer', 'from behind', 'from side',
        # Quality tags
        'highres', 'absurdres', 'high quality', 'detailed',
        'masterpiece', 'best quality',
        # Generic descriptors
        'standing', 'sitting', 'walking',
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantize: bool = True,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: int = 75
    ):
        """
        Initialize the caption enhancer.

        Args:
            model_name: HuggingFace model name for LLM (e.g., "Qwen/Qwen2.5-7B-Instruct")
            device: Device to load model on ("cuda" or "cpu"). If None, auto-detect.
            quantize: Whether to use 4-bit quantisation for LLM
            temperature: Sampling temperature (0.1-2.0, lower = more deterministic)
            top_p: Nucleus sampling threshold (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.quantize = quantize
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # LLM components (lazy-loaded)
        self.model = None
        self.tokenizer = None

        logger.info(f"CaptionEnhancer initialized (device={self.device}, quantize={quantize}, "
                    f"temperature={temperature}, top_p={top_p}, max_tokens={max_new_tokens})")

    def analyse_coverage(
        self,
        tags: List[str],
        description: str
    ) -> Tuple[float, List[str]]:
        """
        Analyse how well the VLM description covers the WD tags.

        Args:
            tags: List of WD tags from tagger
            description: VLM-generated description

        Returns:
            Tuple of (coverage_score, missing_important_tags)
            - coverage_score: 0.0-1.0 indicating % of important tags covered
            - missing_important_tags: List of important tags not in description
        """
        if not tags:
            logger.debug("No tags provided for coverage analysis")
            return 1.0, []

        # Filter tags by importance
        important_tags = self._filter_important_tags(tags)

        if not important_tags:
            logger.debug("No important tags found after filtering")
            return 1.0, []

        # Normalize description for matching
        description_lower = description.lower()

        # Check which tags are covered
        covered_tags = []
        missing_tags = []

        for tag in important_tags:
            tag_lower = tag.lower()

            # Direct substring match
            if tag_lower in description_lower:
                covered_tags.append(tag)
                continue

            # Partial word match (e.g., "long hair" matches "long")
            tag_words = tag_lower.split()
            if any(word in description_lower for word in tag_words if len(word) > 3):
                covered_tags.append(tag)
                continue

            # Tag not found
            missing_tags.append(tag)

        # Calculate coverage score
        coverage_score = len(covered_tags) / len(important_tags) if important_tags else 1.0

        logger.debug(
            f"Coverage analysis: {len(covered_tags)}/{len(important_tags)} important tags covered "
            f"(score={coverage_score:.2f})"
        )

        return coverage_score, missing_tags

    def _filter_important_tags(self, tags: List[str]) -> List[str]:
        """
        Filter tags to keep only important ones for coverage analysis.

        Args:
            tags: All tags from WD Tagger

        Returns:
            Filtered list of important tags
        """
        important_tags = []

        for tag in tags:
            tag_lower = tag.lower()

            # Skip low-priority tags
            if any(low_tag in tag_lower for low_tag in self.LOW_PRIORITY_TAGS):
                continue

            # Keep high-priority tags
            if any(high_tag in tag_lower for high_tag in self.HIGH_PRIORITY_TAGS):
                important_tags.append(tag)
                continue

            # Keep tags with multiple words (usually more descriptive)
            if len(tag.split()) >= 2:
                important_tags.append(tag)

        logger.debug(f"Filtered {len(tags)} tags to {len(important_tags)} important tags")
        return important_tags

    def load_model(self) -> None:
        """
        Lazy-load the LLM model for caption enhancement.

        Raises:
            ValueError: If model_name is not set
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            logger.debug("Model already loaded")
            return

        if not self.model_name:
            raise ValueError("Cannot load model: model_name is not set")

        logger.info(f"Loading LLM model: {self.model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Prepare model loading arguments
            model_kwargs = {
                'device_map': self.device,
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
            }

            # Add quantisation if requested and on CUDA
            if self.quantize and self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs['quantization_config'] = quantization_config
                logger.info("Using 4-bit quantisation")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            logger.info("LLM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise RuntimeError(f"Failed to load LLM model: {e}")

    def enhance_description(
        self,
        original_desc: str,
        missing_tags: List[str],
        coverage: float,
        all_tags: Optional[List[str]] = None
    ) -> str:
        """
        Enhance VLM description using LLM to incorporate missing tags.

        Args:
            original_desc: Original VLM-generated description
            missing_tags: List of important tags not in description
            coverage: Coverage score (0.0-1.0)
            all_tags: All important tags (for heavy rewrite mode)

        Returns:
            Enhanced description with missing tags naturally incorporated
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Determine enhancement strategy based on coverage
        if coverage >= 0.8:
            # No enhancement needed
            logger.debug(f"Coverage {coverage:.2%} >= 80%, skipping enhancement")
            return original_desc

        elif coverage >= 0.5:
            # Light enhancement - add missing details
            prompt = self._generate_light_enhancement_prompt(original_desc, missing_tags)
            logger.debug(f"Using light enhancement (coverage={coverage:.2%})")

        else:
            # Heavy rewrite - incorporate all important tags
            if all_tags is None:
                all_tags = missing_tags
            prompt = self._generate_heavy_rewrite_prompt(original_desc, all_tags)
            logger.debug(f"Using heavy rewrite (coverage={coverage:.2%})")

        # Generate enhanced description
        enhanced_desc = self._generate_from_prompt(prompt)

        return enhanced_desc

    def _generate_light_enhancement_prompt(
        self,
        original_desc: str,
        missing_tags: List[str]
    ) -> str:
        """Generate prompt for light enhancement mode using template from settings."""
        settings = get_settings()
        template = settings.value(
            'enhancement_prompt_light',
            DEFAULT_LIGHT_ENHANCEMENT_PROMPT,
            type=str
        )

        missing_tags_str = ", ".join(missing_tags)

        # Replace template variables
        prompt = template.replace('{original_desc}', original_desc)
        prompt = prompt.replace('{missing_tags}', missing_tags_str)

        return prompt

    def _generate_heavy_rewrite_prompt(
        self,
        original_desc: str,
        all_important_tags: List[str]
    ) -> str:
        """Generate prompt for heavy rewrite mode using template from settings."""
        settings = get_settings()
        template = settings.value(
            'enhancement_prompt_heavy',
            DEFAULT_HEAVY_REWRITE_PROMPT,
            type=str
        )

        all_tags_str = ", ".join(all_important_tags)

        # Replace template variables
        prompt = template.replace('{original_desc}', original_desc)
        prompt = prompt.replace('{all_tags}', all_tags_str)

        return prompt

    def _generate_from_prompt(self, prompt: str) -> str:
        """
        Generate text from LLM given a prompt.

        Args:
            prompt: Input prompt for the LLM

        Returns:
            Generated text
        """
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate using configured parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part (remove prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        # Clean up common LLM artifacts
        generated_text = self._clean_llm_output(generated_text)

        return generated_text

    def _clean_llm_output(self, text: str) -> str:
        """
        Clean up LLM output by removing artifacts and formatting issues.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned text
        """
        # Remove any remaining prompt echoes
        text = re.sub(r'^(Enhanced description:|Description:|Rewritten caption:|Caption:)\s*', '', text, flags=re.IGNORECASE)

        # Remove rule explanations that LLMs sometimes add
        # Pattern: "To meet the rules:" or "Following the rules:" etc.
        text = re.sub(r'\s*(To meet|Following|According to|Based on|Per) the rules?:?.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove numbered rule explanations (e.g., "1. The description..." or "Rule 1:")
        text = re.sub(r'\s*\d+\.\s*(The description|It only|This|I|No|Be|One|Maximum).*$', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\s*Rule \d+:.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove "Note:" or "Notes:" explanations
        text = re.sub(r'\s*Notes?:.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove trailing incomplete sentences
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and not text.rstrip().endswith(('.', '!', '?')):
            # Last sentence is incomplete, remove it
            sentences = sentences[:-1]
        text = '. '.join(sentences)
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def unload_model(self) -> None:
        """
        Unload the LLM model to free memory.
        """
        if self.model is not None:
            logger.info("Unloading LLM model")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # Clear CUDA cache if on GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            logger.info("LLM model unloaded")
