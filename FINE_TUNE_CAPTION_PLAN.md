# Fine-Tune Caption Mode Implementation Plan

## Overview
Add a second output mode to MultiPersonTagger that generates natural-language captions for full model fine-tuning, while preserving the existing LoRA tagging mode.

## Current System Architecture

### Existing LoRA Mode Output
```
person1, tag1, tag2, tag3, person2, tag4, tag5, scene, scene_tag1, scene_tag2
```
Or with custom aliases:
```
singer, tag1, tag2, guitarist, tag3, tag4, scene, scene_tags
```

### Key Components
1. **MultiPersonTagger** (`auto_captioning/models/multi_person_tagger.py`)
   - Person detection via YOLOv8
   - Per-person WD tagging
   - Scene tagging
   - Person alias assignment
   - Tag formatting and output

2. **WD Tagger** (`auto_captioning/models/wd_tagger.py`)
   - Generates comma-separated tags
   - Returns raw tag lists

## New Fine-Tune Mode Output

### Target Format
```
sksA: A young woman with long brown hair wearing a blue dress and pearl necklace, smiling at the camera in a confident pose.
sksB: An older man in a black suit and red tie, standing with arms crossed looking serious.
Scene: Indoor corporate office with white walls and large windows.
```

### Alias Assignment Rules
- First detected person: `sksA`
- Second detected person: `sksB`
- Third+ detected: `"unidentified man/woman/person"` (based on tags)

### Natural Language Requirements
- 15-35 words per person
- Grammatically correct sentences
- Clear subject-verb-object structure
- Clothing, accessories, actions associated with correct person
- Filter out quality/rating/metadata tags

## Implementation Plan

### Phase 1: Add Mode Parameter to MultiPersonTagger

#### 1.1 Update MultiPersonTagger.__init__()
**File**: `auto_captioning/models/multi_person_tagger.py`

Add parameter:
```python
def __init__(self,
             model_id: str,
             caption_mode: str = 'lora_tags',  # NEW: 'lora_tags' or 'fine_tune_caption'
             person_aliases: Optional[List[str]] = None,
             ...):
    self.caption_mode = caption_mode
```

#### 1.2 Store mode in settings
Ensure mode is passed through from UI settings to model initialization.

### Phase 2: Tag Filtering System

#### 2.1 Create Tag Filter Lists
**File**: `auto_captioning/models/multi_person_tagger.py`

Add module-level constants:
```python
# Tags to exclude from fine-tune captions
EXCLUDED_TAG_CATEGORIES = {
    'quality': [
        'masterpiece', 'best quality', 'high quality', 'normal quality',
        'low quality', 'worst quality', 'highres', 'absurdres',
        'incredibly absurdres', 'huge filesize', 'wallpaper'
    ],
    'rating': [
        'rating:safe', 'rating:questionable', 'rating:explicit',
        'rating: safe', 'rating: questionable', 'rating: explicit',
        'safe', 'questionable', 'explicit', 'nsfw', 'sfw'
    ],
    'metadata': [
        'sketch', 'lineart', 'monochrome', 'greyscale', 'comic',
        'traditional media', 'pixel art', 'photo', 'realistic',
        'translation request', 'commentary', 'english commentary'
    ],
    'structural': [
        'solo', '1girl', '1boy', '2girls', '2boys', '3girls', '3boys',
        'multiple girls', 'multiple boys', 'multiple people',
        'focus', 'male focus', 'female focus', 'censored', 'uncensored'
    ]
}

def should_exclude_tag(tag: str) -> bool:
    """Check if tag should be excluded from fine-tune captions."""
    tag_lower = tag.lower().strip()
    for category, tags in EXCLUDED_TAG_CATEGORIES.items():
        if tag_lower in tags:
            return True
    return False
```

### Phase 3: Tag-to-Natural-Language Converter

#### 3.1 Create Conversion Function
**File**: `auto_captioning/models/multi_person_tagger.py`

```python
def tags_to_natural_language(
    tags: List[str],
    person_alias: str,
    target_word_count: tuple = (15, 35)
) -> str:
    """
    Convert WD tags into a natural language sentence.

    Args:
        tags: List of WD tags for a person
        person_alias: The person's alias (e.g., "sksA", "sksB")
        target_word_count: Min/max words (default 15-35)

    Returns:
        Natural language description as a single sentence
    """
    # Filter out excluded tags
    filtered_tags = [tag for tag in tags if not should_exclude_tag(tag)]

    if not filtered_tags:
        return f"{person_alias}: A person in the image."

    # Categorize tags by type
    categories = categorize_tags(filtered_tags)

    # Build sentence structure
    sentence = build_sentence_from_categories(categories, person_alias)

    # Ensure word count is within range
    sentence = adjust_sentence_length(sentence, target_word_count)

    return sentence
```

#### 3.2 Tag Categorization
```python
def categorize_tags(tags: List[str]) -> dict:
    """
    Categorize tags into semantic groups.

    Returns dict with keys:
    - 'person_type': ['girl', 'woman', 'boy', 'man']
    - 'appearance': ['long hair', 'brown hair', 'blue eyes']
    - 'clothing': ['dress', 'shirt', 'pants']
    - 'accessories': ['necklace', 'hat', 'glasses']
    - 'pose': ['standing', 'sitting', 'arms crossed']
    - 'expression': ['smiling', 'serious', 'laughing']
    - 'action': ['looking at viewer', 'waving', 'holding object']
    - 'other': [remaining tags]
    """
    # Implementation using keyword matching
    # E.g., tags ending in 'hair' go to appearance
    # Tags ending in 'wear', 'dress', 'shirt' go to clothing
    # etc.
```

#### 3.3 Sentence Builder
```python
def build_sentence_from_categories(categories: dict, person_alias: str) -> str:
    """
    Build a grammatically correct sentence from categorized tags.

    Structure:
    "A [person_type] with [appearance] wearing [clothing] and [accessories],
     [expression] while [action/pose]."
    """
    parts = []

    # Start with person type + appearance
    person_type = categories.get('person_type', ['person'])[0]
    appearance = categories.get('appearance', [])

    if appearance:
        parts.append(f"A {person_type} with {' and '.join(appearance[:3])}")
    else:
        parts.append(f"A {person_type}")

    # Add clothing
    clothing = categories.get('clothing', [])
    if clothing:
        parts.append(f"wearing {format_list_naturally(clothing)}")

    # Add accessories
    accessories = categories.get('accessories', [])
    if accessories:
        parts.append(f"and {format_list_naturally(accessories)}")

    # Add expression/action
    expression = categories.get('expression', [])
    action = categories.get('action', [])
    pose = categories.get('pose', [])

    activity_parts = []
    if expression:
        activity_parts.append(expression[0])
    if action:
        activity_parts.append(action[0])
    elif pose:
        activity_parts.append(pose[0])

    if activity_parts:
        parts.append(', '.join(activity_parts))

    # Join with proper punctuation
    sentence = ' '.join(parts) + '.'

    # Capitalize first letter
    sentence = sentence[0].upper() + sentence[1:]

    return sentence
```

### Phase 4: Stable Alias Assignment

#### 4.1 Update Alias Logic
**File**: `auto_captioning/models/multi_person_tagger.py`

Modify existing alias assignment to support fine-tune mode:
```python
def assign_person_aliases(
    self,
    num_people: int,
    custom_aliases: Optional[List[str]] = None,
    detected_genders: Optional[List[str]] = None
) -> List[str]:
    """
    Assign aliases to detected people based on mode.

    Args:
        num_people: Number of detected people
        custom_aliases: User-provided aliases (if any)
        detected_genders: Gender hints from tags ['female', 'male', 'unknown']

    Returns:
        List of aliases (e.g., ['sksA', 'sksB', 'unidentified woman'])
    """
    if custom_aliases and len(custom_aliases) >= num_people:
        return custom_aliases[:num_people]

    if self.caption_mode == 'fine_tune_caption':
        # Fine-tune mode uses sksA, sksB, then unidentified
        aliases = []
        for i in range(num_people):
            if i == 0:
                aliases.append('sksA')
            elif i == 1:
                aliases.append('sksB')
            else:
                # Third+ person: use gender-specific unidentified
                gender = detected_genders[i] if detected_genders else 'unknown'
                if gender == 'female':
                    aliases.append('unidentified woman')
                elif gender == 'male':
                    aliases.append('unidentified man')
                else:
                    aliases.append('unidentified person')
        return aliases
    else:
        # LoRA mode uses person1, person2, etc. or custom aliases
        if custom_aliases:
            # Extend with default person labels if needed
            aliases = list(custom_aliases)
            for i in range(len(aliases), num_people):
                aliases.append(f'person{i+1}')
            return aliases[:num_people]
        else:
            return [f'person{i+1}' for i in range(num_people)]
```

### Phase 5: Output Formatting

#### 5.1 Modify generate_caption() Method
**File**: `auto_captioning/models/multi_person_tagger.py`

Branch based on caption_mode:
```python
def generate_caption(self, image_path: str, ...) -> str:
    """Generate caption based on current mode."""

    # ... existing detection logic ...
    # (person detection, tag generation, scene tagging all stay the same)

    # Branch based on mode
    if self.caption_mode == 'fine_tune_caption':
        return self._format_fine_tune_caption(person_data, scene_tags)
    else:  # 'lora_tags'
        return self._format_lora_tags(person_data, scene_tags)
```

#### 5.2 Create Fine-Tune Formatter
```python
def _format_fine_tune_caption(
    self,
    person_data: List[dict],
    scene_tags: List[str]
) -> str:
    """
    Format output for fine-tune mode.

    person_data format:
    [
        {'alias': 'sksA', 'tags': ['tag1', 'tag2', ...]},
        {'alias': 'sksB', 'tags': ['tag3', 'tag4', ...]},
    ]

    Returns multi-line caption:
    sksA: [natural language description]
    sksB: [natural language description]
    Scene: [optional scene description]
    """
    lines = []

    # Generate description for each person
    for person in person_data:
        alias = person['alias']
        tags = person['tags']
        description = tags_to_natural_language(tags, alias)
        lines.append(f"{alias}: {description}")

    # Add scene description if present
    if scene_tags:
        scene_description = scene_tags_to_natural_language(scene_tags)
        if scene_description:
            lines.append(f"Scene: {scene_description}")

    return '\n'.join(lines)
```

#### 5.3 Scene Description Converter
```python
def scene_tags_to_natural_language(scene_tags: List[str]) -> str:
    """
    Convert scene tags to brief natural language description.
    Target: 5-15 words
    """
    # Filter out excluded tags
    filtered = [tag for tag in scene_tags if not should_exclude_tag(tag)]

    if not filtered:
        return ""

    # Simple concatenation for scene (more permissive than person descriptions)
    # E.g., "indoor, office, windows, desk" -> "Indoor office with windows and desk"

    # Detect location type
    location_tags = [tag for tag in filtered if tag in ['indoor', 'outdoor', 'studio']]
    setting_tags = [tag for tag in filtered if tag not in location_tags]

    parts = []
    if location_tags:
        parts.append(location_tags[0].capitalize())

    if setting_tags:
        # Take first few relevant tags
        parts.extend(setting_tags[:4])

    if len(parts) <= 1:
        return ' '.join(parts)
    else:
        # "Indoor office with windows and desk"
        return f"{parts[0]} {parts[1]}" + (f" with {format_list_naturally(parts[2:])}" if len(parts) > 2 else "")
```

### Phase 6: UI Integration

#### 6.1 Add Mode Selector to AutoCaptioner
**File**: `taggui/widgets/auto_captioner.py`

In the MultiPersonTagger settings section, add:
```python
# Caption mode selector
caption_mode_label = QLabel("Caption Mode:")
self.caption_mode_combo = QComboBox()
self.caption_mode_combo.addItems(['LoRA Tags', 'Fine-Tune Caption'])
self.caption_mode_combo.setCurrentIndex(0)  # Default to LoRA
self.caption_mode_combo.setToolTip(
    "LoRA Tags: Comma-separated tags with person labels\n"
    "Fine-Tune Caption: Natural language descriptions with sksA/sksB aliases"
)
# Add to form layout
```

#### 6.2 Pass Mode to Model
When initializing MultiPersonTagger:
```python
caption_mode = 'fine_tune_caption' if self.caption_mode_combo.currentText() == 'Fine-Tune Caption' else 'lora_tags'

model = MultiPersonTagger(
    model_id=model_id,
    caption_mode=caption_mode,  # NEW parameter
    person_aliases=person_aliases,
    ...
)
```

### Phase 7: Helper Functions

#### 7.1 Natural List Formatting
```python
def format_list_naturally(items: List[str]) -> str:
    """
    Format list naturally: "a, b and c" or "a and b" or "a"
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ', '.join(items[:-1]) + f" and {items[-1]}"
```

#### 7.2 Word Count Adjustment
```python
def adjust_sentence_length(sentence: str, target_range: tuple) -> str:
    """
    Adjust sentence to fit within target word count range.
    If too long: remove less important modifiers
    If too short: keep as is (we won't pad artificially)
    """
    words = sentence.split()
    word_count = len(words)
    min_words, max_words = target_range

    if word_count > max_words:
        # Trim to max words, keeping sentence structure
        # Try to cut at natural boundaries (commas, 'and', etc.)
        return smart_trim_sentence(sentence, max_words)

    # Don't artificially pad short sentences
    return sentence
```

## File Structure Summary

### Files to Modify
1. **`auto_captioning/models/multi_person_tagger.py`** (MAIN CHANGES)
   - Add `caption_mode` parameter
   - Add tag filtering constants
   - Add `should_exclude_tag()`
   - Add `tags_to_natural_language()`
   - Add `categorize_tags()`
   - Add `build_sentence_from_categories()`
   - Add `scene_tags_to_natural_language()`
   - Add `format_list_naturally()`
   - Modify `assign_person_aliases()` to support sksA/sksB
   - Add `_format_fine_tune_caption()`
   - Modify `generate_caption()` to branch on mode

2. **`taggui/widgets/auto_captioner.py`** (UI CHANGES)
   - Add caption mode selector combo box
   - Pass mode parameter to MultiPersonTagger

### No Changes Required
- `auto_captioning/models/wd_tagger.py` - stays the same
- Person detection logic - stays the same
- YOLOv8 integration - stays the same

## Testing Strategy

### Test Cases
1. **Single person image** → Should output only `sksA: ...`
2. **Two person image** → Should output `sksA: ...` and `sksB: ...`
3. **Three+ person image** → sksA, sksB, unidentified person
4. **Scene tags** → Optional Scene line at end
5. **Quality tags** → Should be filtered out
6. **Word count** → Verify 15-35 words per person
7. **Mode switching** → Toggle between LoRA and Fine-Tune modes
8. **Custom aliases** → Verify they're only used in LoRA mode

## Implementation Order

1. ✅ Create this plan document
2. Add tag filtering (constants + function)
3. Add tag categorization function
4. Add sentence builder function
5. Add natural language converter
6. Update alias assignment logic
7. Add fine-tune formatter
8. Modify generate_caption() to branch on mode
9. Add UI mode selector
10. Test with sample images
11. Refine natural language quality

## Key Design Decisions

### Why Not Use LLM/AI for Tag Conversion?
- Requirement: "Use only Python standard libraries"
- Rule-based conversion is deterministic and fast
- Pattern matching sufficient for WD tag structure

### Alias Naming Convention
- `sksA`, `sksB`: Common fine-tuning convention (from Stable Diffusion community)
- Not `person1`, `person2`: Those are for LoRA training
- `unidentified X`: Clarifies tertiary subjects

### Colon vs Comma After Alias
- Fine-tune: `sksA: description` (formal structure)
- LoRA: `person1, tags, person2, tags` (tag list)

### Tag-to-Sentence Quality
- Priority: Grammatical correctness > completeness
- Better to have shorter clear sentence than awkward long one
- User can always switch back to LoRA mode for full tags

## Open Questions / Decisions Needed

1. **Scene line**: Always include or only if meaningful tags exist?
   - **Recommendation**: Only include if filtered scene tags exist

2. **Gender detection**: How to determine for "unidentified X"?
   - **Recommendation**: Parse WD tags for gender indicators ('girl', 'boy', 'man', 'woman')

3. **Word count enforcement**: Strict or flexible?
   - **Recommendation**: Flexible - use 15-35 as target but allow natural variation

4. **Punctuation in sentences**: Multiple sentences or single long sentence?
   - **Recommendation**: Single sentence per person for consistency

5. **Tag priority**: Which tags are most important when building sentence?
   - **Recommendation**: appearance > clothing > accessories > action > other

## Example Outputs

### Input Tags (LoRA mode output)
```
person1, 1girl, long hair, brown hair, blue dress, necklace, smiling, looking at viewer,
person2, 1boy, short hair, black suit, red tie, arms crossed, serious,
scene, indoor, office, windows, desk
```

### Output (Fine-Tune mode)
```
sksA: A young woman with long brown hair wearing a blue dress and necklace, smiling while looking at the viewer.
sksB: A man with short hair wearing a black suit and red tie, standing with arms crossed looking serious.
Scene: Indoor office with windows and desk.
```

---

**Status**: Plan complete, awaiting approval before implementation
