# Fine-Tune Caption Mode Implementation Plan (V2 - VLM-Based)

## Overview
Add a second output mode to MultiPersonTagger that generates natural-language captions using VLM models to describe masked person regions, producing output suitable for full model fine-tuning.

## Key Insight
Instead of converting WD tags to natural language (brittle, limited), we use existing VLM models to directly describe each person's masked/cropped region. This produces:
- ✅ Natural, fluent descriptions
- ✅ Better context understanding
- ✅ Can describe things WD tags miss
- ✅ Leverages existing model infrastructure

## Architecture Overview

### Current LoRA Mode (Tag-Based)
```
Detection → WD Tagger per person → Format as tags → Output
person1, tag1, tag2, person2, tag3, tag4, scene, scene_tags
```

### New Fine-Tune Mode (VLM-Based)
```
Detection → Crop each person → VLM describe → Format with aliases → Output
sksA: [VLM natural description]
sksB: [VLM natural description]
Scene: [VLM scene description]
```

## Implementation Plan

### Phase 1: Add VLM Description Method to MultiPersonTagger

#### 1.1 Add Caption Mode Parameter
**File**: `auto_captioning/models/multi_person_tagger.py`

```python
def __init__(self,
             model_id: str,
             caption_mode: str = 'lora_tags',  # 'lora_tags' or 'fine_tune_caption'
             description_model: str = 'moondream2',  # VLM for descriptions
             person_aliases: Optional[List[str]] = None,
             ...):
    self.caption_mode = caption_mode
    self.description_model_name = description_model
    self.description_model = None  # Lazy loaded
```

#### 1.2 Lazy-Load Description Model
```python
def _load_description_model(self):
    """Lazy-load the VLM model for natural language descriptions."""
    if self.description_model is not None:
        return

    # Import and instantiate the appropriate model
    from auto_captioning.models_list import get_model_class

    model_class = get_model_class(self.description_model_name)
    self.description_model = model_class(self.description_model_name)

    logger.info(f"Loaded description model: {self.description_model_name}")
```

### Phase 2: Person Region Description

#### 2.1 Add Method to Describe Person Region
**File**: `auto_captioning/models/multi_person_tagger.py`

```python
def describe_person_region(
    self,
    image: PilImage,
    mask: np.ndarray,
    bbox: tuple,
    person_index: int
) -> str:
    """
    Generate natural language description of a person region using VLM.

    Args:
        image: Full PIL image
        mask: Boolean mask for this person (or None)
        bbox: Bounding box (x1, y1, x2, y2)
        person_index: Index of person (for alias assignment)

    Returns:
        Natural language description (15-35 words)
    """
    # Ensure description model is loaded
    self._load_description_model()

    # Crop to person region with padding
    x1, y1, x2, y2 = bbox
    padding = 20  # Small padding for context
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    cropped = image.crop((x1, y1, x2, y2))

    # Optional: Apply mask to focus on person (create RGBA with transparency)
    if mask is not None:
        cropped = apply_mask_to_crop(cropped, mask, bbox)

    # Craft prompt for VLM
    prompt = (
        "Describe this person in one clear sentence (15-35 words). "
        "Include: physical appearance, clothing, accessories, pose, and expression. "
        "Be specific and concise. Do not use bullet points."
    )

    # Generate description using VLM
    try:
        # Save cropped region to temp file (most VLMs expect file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cropped.save(tmp.name)
            tmp_path = tmp.name

        # Call VLM (using existing interface)
        description = self.description_model.generate(
            image_path=tmp_path,
            prompt=prompt,
            max_new_tokens=100,  # ~35 words
            temperature=0.7,
            top_p=0.9
        )

        # Clean up temp file
        import os
        os.unlink(tmp_path)

        # Clean up description (remove prompt echo, trim whitespace)
        description = clean_vlm_output(description, prompt)

        logger.debug(f"Person {person_index} description: {description}")
        return description

    except Exception as e:
        logger.error(f"Failed to generate description for person {person_index}: {e}")
        return "A person in the image."
```

#### 2.2 Apply Mask to Crop (Optional Enhancement)
```python
def apply_mask_to_crop(
    cropped: PilImage,
    mask: np.ndarray,
    bbox: tuple
) -> PilImage:
    """
    Apply mask to cropped region, making background semi-transparent.
    This helps VLM focus on the person.
    """
    x1, y1, x2, y2 = bbox

    # Extract mask region corresponding to crop
    mask_crop = mask[y1:y2, x1:x2]

    # Resize mask to match crop size (in case of padding differences)
    from PIL import Image as PilImage
    mask_pil = PilImage.fromarray((mask_crop * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(cropped.size, PilImage.LANCZOS)

    # Convert crop to RGBA
    cropped_rgba = cropped.convert('RGBA')

    # Apply mask as alpha channel (or dim background)
    # Option 1: Full transparency (might confuse some VLMs)
    # Option 2: Dim background (better - keep context)
    mask_array = np.array(mask_pil)
    alpha = np.where(mask_array > 128, 255, 80)  # Person=255, bg=80 (dimmed)

    # Apply alpha
    rgba_array = np.array(cropped_rgba)
    rgba_array[:, :, 3] = alpha

    return PilImage.fromarray(rgba_array)
```

#### 2.3 Clean VLM Output
```python
def clean_vlm_output(description: str, prompt: str) -> str:
    """
    Clean up VLM output by removing:
    - Prompt echo
    - Extra whitespace
    - Unwanted formatting
    - Ensure single sentence
    """
    # Remove prompt if echoed
    if description.startswith(prompt):
        description = description[len(prompt):].strip()

    # Remove common prefixes that VLMs add
    prefixes_to_remove = [
        "The person is ",
        "This person is ",
        "The image shows ",
        "In the image, ",
        "Description: ",
        "A: ",
        "Answer: "
    ]
    for prefix in prefixes_to_remove:
        if description.lower().startswith(prefix.lower()):
            description = description[len(prefix):].strip()

    # Ensure starts with capital
    if description:
        description = description[0].upper() + description[1:]

    # Ensure ends with period
    if description and not description.endswith('.'):
        description += '.'

    # Remove multiple spaces
    description = ' '.join(description.split())

    return description
```

### Phase 3: Scene Description

#### 3.1 Generate Scene Description
```python
def describe_scene(
    self,
    image: PilImage,
    person_masks: List[np.ndarray] = None
) -> str:
    """
    Generate brief scene/setting description.

    Args:
        image: Full PIL image
        person_masks: Optional list of person masks to exclude

    Returns:
        Brief scene description (5-15 words) or empty string
    """
    self._load_description_model()

    # Optional: Create version with people masked out
    scene_image = image
    if person_masks:
        scene_image = mask_out_people(image, person_masks)

    prompt = (
        "Briefly describe the setting/environment in 5-15 words. "
        "Focus on location type, lighting, and key background elements. "
        "Do not describe people."
    )

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            scene_image.save(tmp.name)
            tmp_path = tmp.name

        description = self.description_model.generate(
            image_path=tmp_path,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )

        import os
        os.unlink(tmp_path)

        description = clean_vlm_output(description, prompt)

        # Only return if substantial description (filter out generic responses)
        if len(description.split()) >= 3:
            return description
        return ""

    except Exception as e:
        logger.error(f"Failed to generate scene description: {e}")
        return ""
```

#### 3.2 Mask Out People (Optional)
```python
def mask_out_people(image: PilImage, person_masks: List[np.ndarray]) -> PilImage:
    """
    Create version of image with people regions masked out (greyed/blurred).
    This helps VLM focus on background/setting.
    """
    img_array = np.array(image)

    # Combine all person masks
    combined_mask = np.zeros(img_array.shape[:2], dtype=bool)
    for mask in person_masks:
        combined_mask |= mask

    # Apply greyscale or blur to person regions
    from PIL import ImageFilter
    result = image.copy()

    # Option 1: Grey out people
    grey_array = img_array.copy()
    grey_array[combined_mask] = grey_array[combined_mask] * 0.3  # Dim to 30%

    # Option 2: Blur people (might be better for context)
    # result = result.filter(ImageFilter.GaussianBlur(10))

    return PilImage.fromarray(grey_array.astype(np.uint8))
```

### Phase 4: Alias Assignment with Gender Detection

#### 4.1 Detect Gender from Tags or VLM
```python
def detect_gender_from_tags(tags: List[str]) -> str:
    """
    Detect gender from WD tags.

    Returns: 'female', 'male', or 'unknown'
    """
    female_indicators = ['girl', 'woman', 'female', 'feminine']
    male_indicators = ['boy', 'man', 'male', 'masculine']

    tags_lower = [tag.lower() for tag in tags]

    # Check female indicators
    if any(indicator in ' '.join(tags_lower) for indicator in female_indicators):
        return 'female'

    # Check male indicators
    if any(indicator in ' '.join(tags_lower) for indicator in male_indicators):
        return 'male'

    return 'unknown'
```

#### 4.2 Assign Aliases Based on Mode
```python
def assign_person_aliases(
    self,
    num_people: int,
    person_tags: List[List[str]] = None,
    custom_aliases: Optional[List[str]] = None
) -> List[str]:
    """
    Assign aliases based on caption mode.

    Args:
        num_people: Number of detected people
        person_tags: Tags for each person (for gender detection)
        custom_aliases: User-provided custom aliases

    Returns:
        List of aliases
    """
    # Custom aliases always take precedence
    if custom_aliases and len(custom_aliases) >= num_people:
        return custom_aliases[:num_people]

    if self.caption_mode == 'fine_tune_caption':
        # Fine-tune mode: sksA, sksB, unidentified X
        aliases = []
        for i in range(num_people):
            if i == 0:
                aliases.append('sksA')
            elif i == 1:
                aliases.append('sksB')
            else:
                # Detect gender for unidentified label
                gender = 'unknown'
                if person_tags and i < len(person_tags):
                    gender = detect_gender_from_tags(person_tags[i])

                if gender == 'female':
                    aliases.append('unidentified woman')
                elif gender == 'male':
                    aliases.append('unidentified man')
                else:
                    aliases.append('unidentified person')
        return aliases
    else:
        # LoRA mode: person1, person2, etc.
        if custom_aliases:
            aliases = list(custom_aliases)
            for i in range(len(aliases), num_people):
                aliases.append(f'person{i+1}')
            return aliases[:num_people]
        else:
            return [f'person{i+1}' for i in range(num_people)]
```

### Phase 5: Modified generate_caption() Flow

#### 5.1 Branch on Caption Mode
**File**: `auto_captioning/models/multi_person_tagger.py`

```python
def generate_caption(
    self,
    image_path: str,
    prompt: str,
    use_cpu: bool = False,
    **kwargs
) -> str:
    """
    Generate caption based on caption_mode.

    Flow:
    1. Detect people (same for both modes)
    2. Branch based on mode:
       - LoRA: Generate WD tags per person
       - Fine-tune: Generate VLM descriptions per person
    3. Format and return
    """
    # Load image
    image = PilImage.open(image_path).convert('RGB')

    # Detect people (YOLOv8)
    detections = self.detect_people(image)

    if not detections:
        return "No people detected in image."

    # Branch based on mode
    if self.caption_mode == 'fine_tune_caption':
        return self._generate_fine_tune_caption(image, detections)
    else:
        return self._generate_lora_tags(image, detections)
```

#### 5.2 Fine-Tune Caption Generator
```python
def _generate_fine_tune_caption(
    self,
    image: PilImage,
    detections: List[dict]
) -> str:
    """
    Generate fine-tune style caption using VLM descriptions.

    detections format:
    [
        {'bbox': (x1,y1,x2,y2), 'mask': np.ndarray, 'confidence': float},
        ...
    ]

    Returns multi-line format:
    sksA: [VLM description]
    sksB: [VLM description]
    Scene: [VLM scene description]
    """
    num_people = len(detections)

    # First, get tags for gender detection (fast WD tagger pass)
    # This is optional - we could skip and just use 'person' for all
    person_tags = []
    if num_people > 2:  # Only needed for 3+ people
        for detection in detections:
            bbox = detection['bbox']
            cropped = image.crop(bbox)
            # Quick WD tag (just for gender detection)
            tags, _ = self.wd_model.generate_tags(
                np.array(cropped),
                general_threshold=0.5,
                character_threshold=0.5
            )
            person_tags.append(tags)

    # Assign aliases
    aliases = self.assign_person_aliases(
        num_people=num_people,
        person_tags=person_tags if person_tags else None,
        custom_aliases=self.person_aliases
    )

    # Generate VLM description for each person
    lines = []
    person_masks = []

    for i, detection in enumerate(detections):
        alias = aliases[i]
        bbox = detection['bbox']
        mask = detection.get('mask')

        # Generate natural language description
        description = self.describe_person_region(
            image=image,
            mask=mask,
            bbox=bbox,
            person_index=i
        )

        lines.append(f"{alias}: {description}")

        if mask is not None:
            person_masks.append(mask)

    # Generate scene description
    scene_desc = self.describe_scene(image, person_masks)
    if scene_desc:
        lines.append(f"Scene: {scene_desc}")

    return '\n'.join(lines)
```

#### 5.3 LoRA Tags Generator (Existing, Unchanged)
```python
def _generate_lora_tags(
    self,
    image: PilImage,
    detections: List[dict]
) -> str:
    """
    Generate LoRA-style tags (existing logic, no changes).

    Returns: person1, tag1, tag2, person2, tag3, scene, scene_tags
    """
    # ... existing implementation ...
    # (No changes - this is the current MultiPersonTagger logic)
```

### Phase 6: UI Integration

#### 6.1 Add Mode Selector to AutoCaptioner
**File**: `taggui/widgets/auto_captioner.py`

In the MultiPersonTagger settings section (around line 3300+):

```python
# Caption Mode selector
caption_mode_label = QLabel("Caption Mode:")
self.mpt_caption_mode_combo = QComboBox()
self.mpt_caption_mode_combo.addItems(['LoRA Tags', 'Fine-Tune Caption'])
self.mpt_caption_mode_combo.setCurrentIndex(0)
self.mpt_caption_mode_combo.setToolTip(
    "LoRA Tags: Comma-separated WD tags with person labels (fast)\n"
    "Fine-Tune Caption: Natural language descriptions using VLM (slower but higher quality)"
)
multi_person_settings_layout.addRow(caption_mode_label, self.mpt_caption_mode_combo)

# Description Model selector (for fine-tune mode)
desc_model_label = QLabel("Description Model:")
self.mpt_desc_model_combo = QComboBox()
# Add VLM models suitable for description
self.mpt_desc_model_combo.addItems([
    'moondream2',           # Fast, good quality
    'Florence-2-base',      # Very fast
    'Florence-2-large',     # Better quality
    'llava-v1.6-vicuna-7b', # High quality but slower
])
self.mpt_desc_model_combo.setCurrentIndex(0)  # Default to moondream
self.mpt_desc_model_combo.setToolTip(
    "VLM model used to generate natural language descriptions.\n"
    "Moondream2: Fast and good quality (recommended)\n"
    "Florence-2: Faster but shorter descriptions\n"
    "LLaVA: Highest quality but slower"
)
multi_person_settings_layout.addRow(desc_model_label, self.mpt_desc_model_combo)

# Enable/disable description model selector based on mode
def on_caption_mode_changed():
    is_fine_tune = self.mpt_caption_mode_combo.currentText() == 'Fine-Tune Caption'
    self.mpt_desc_model_combo.setEnabled(is_fine_tune)
    desc_model_label.setEnabled(is_fine_tune)

self.mpt_caption_mode_combo.currentTextChanged.connect(on_caption_mode_changed)
on_caption_mode_changed()  # Initial state
```

#### 6.2 Pass Parameters to Model
When initializing MultiPersonTagger (around line with `model = MultiPersonTagger(...)`):

```python
# Get caption mode
caption_mode = 'fine_tune_caption' if self.mpt_caption_mode_combo.currentText() == 'Fine-Tune Caption' else 'lora_tags'

# Get description model (if fine-tune mode)
description_model = self.mpt_desc_model_combo.currentText() if caption_mode == 'fine_tune_caption' else 'moondream2'

# Initialize model
model = MultiPersonTagger(
    model_id=model_id,
    caption_mode=caption_mode,           # NEW
    description_model=description_model, # NEW
    person_aliases=person_aliases,
    use_cpu=use_cpu,
    device=device,
    ...
)
```

### Phase 7: Performance Considerations

#### 7.1 Model Caching
- Description model is lazy-loaded and cached
- Only loaded once per captioning session
- Reused across multiple images

#### 7.2 Batch Processing Optimization
For multiple images:
```python
# Load model once
model._load_description_model()

# Process multiple images
for image_path in image_paths:
    caption = model.generate_caption(image_path, ...)
    # Model already loaded, no reload overhead
```

#### 7.3 Speed Estimates
- LoRA mode: ~1-2s per image (WD tagger only)
- Fine-tune mode: ~5-10s per image (VLM description)
  - YOLOv8 detection: ~0.5s
  - Per-person VLM: ~2-3s each
  - Scene VLM: ~2-3s
  - Total: ~5s for 1 person, ~10s for 2 people

### Phase 8: Testing Strategy

#### Test Cases
1. **Single person** → `sksA: [description]`
2. **Two people** → `sksA: ...` + `sksB: ...`
3. **Three+ people** → sksA, sksB, unidentified
4. **Scene description** → Optional Scene line
5. **Mode switching** → Toggle UI and verify output changes
6. **Different VLM models** → Test quality/speed tradeoffs
7. **Word count** → Verify descriptions are 15-35 words
8. **Custom aliases** → Verify LoRA mode still uses them

#### Sample Test Images
- Portrait (1 person)
- Group photo (2-3 people)
- Crowd scene (4+ people)
- Various settings (indoor/outdoor)

## File Summary

### Files to Modify

1. **`auto_captioning/models/multi_person_tagger.py`** (MAJOR)
   - Add `caption_mode` and `description_model` parameters to `__init__()`
   - Add `_load_description_model()`
   - Add `describe_person_region()`
   - Add `describe_scene()`
   - Add `apply_mask_to_crop()`
   - Add `clean_vlm_output()`
   - Add `detect_gender_from_tags()`
   - Modify `assign_person_aliases()` to support sksA/sksB
   - Modify `generate_caption()` to branch on mode
   - Add `_generate_fine_tune_caption()`
   - Rename existing caption logic to `_generate_lora_tags()`

2. **`taggui/widgets/auto_captioner.py`** (MINOR)
   - Add caption mode combo box
   - Add description model combo box
   - Add mode change handler
   - Pass parameters to MultiPersonTagger

### No Changes
- `auto_captioning/models/wd_tagger.py`
- YOLOv8 detection logic
- Other VLM model files
- Image processing utils

## Advantages of VLM Approach

✅ **Natural Language**: Real VLM-generated descriptions, not converted tags
✅ **Better Quality**: Understands context, relationships, nuance
✅ **Flexible**: Can describe anything the VLM can see
✅ **Maintainable**: No complex rule-based conversion
✅ **Consistent**: Same model architecture as other captioning
✅ **Extensible**: Easy to swap VLM models or adjust prompts

## Trade-offs

⚠️ **Speed**: Slower than WD tags (5-10s vs 1-2s per image)
⚠️ **Determinism**: VLM output may vary slightly between runs
⚠️ **Quality Variance**: Depends on chosen VLM model
⚠️ **Memory**: Need to load both WD tagger AND VLM

### Mitigation
- User can choose fast model (Moondream2) vs quality (LLaVA)
- Batch processing reuses loaded model
- LoRA mode still available for speed-critical workflows

## Example Outputs

### Input Image
- Person 1: Woman in blue dress with long brown hair, smiling
- Person 2: Man in black suit with short grey hair, serious
- Setting: Modern office with large windows

### Output (Fine-Tune Mode with Moondream2)
```
sksA: A young woman with long brown hair wearing an elegant blue dress and pearl necklace, smiling warmly at the camera with a confident expression.
sksB: A middle-aged man with short grey hair dressed in a formal black suit and red tie, standing with arms crossed and a serious professional demeanor.
Scene: Modern corporate office interior with large floor-to-ceiling windows and minimalist white furniture.
```

### Output (LoRA Mode - Unchanged)
```
person1, 1girl, long hair, brown hair, blue dress, necklace, smile, looking at viewer, person2, 1boy, short hair, grey hair, black suit, red tie, arms crossed, serious, scene, indoor, office, window, desk, corporate
```

## Open Questions

1. **Scene description**: Include always or make optional?
   - **Recommendation**: Always attempt, only include if meaningful (3+ words)

2. **Mask application**: Should we mask background when describing people?
   - **Recommendation**: Yes, dim background to 30% to help VLM focus

3. **Default VLM**: Which model for best speed/quality balance?
   - **Recommendation**: Moondream2 (fast, good quality, small memory footprint)

4. **Prompt engineering**: Should prompts be user-configurable?
   - **Recommendation**: No - keep hardcoded for consistency. Can add advanced option later.

5. **Fallback**: What if VLM fails to generate description?
   - **Recommendation**: Return generic "A person in the image." and log error

6. **Word count enforcement**: Should we trim VLM output to 35 words?
   - **Recommendation**: Yes - truncate at sentence boundary if over 35 words

---

**Status**: Plan V2 complete (VLM-based approach), awaiting approval before implementation
