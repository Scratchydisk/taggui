# Enhanced Fine-Tune Caption Generation Design

## Overview

This document specifies the design for enhancing fine-tune caption generation quality in TagGUI. The current system produces generic descriptions that miss tag-level detail and scene descriptions that incorrectly mention people. This enhancement introduces two modes: a standard mode for VRAM-constrained systems and an LLM-enhanced mode for higher quality output.

## Current Problems

### Problem 1: Generic Person Descriptions
- VLMs (Vision Language Models) produce conservative, generic descriptions
- Important details from tags are missed (e.g., tags say "long black hair, blue eyes, red dress" but VLM says "a woman in formal attire")
- VLMs lack context about the level of detail needed for AI training datasets
- Trade-off between natural language flow and comprehensive specificity

### Problem 2: Scene Contamination
- Scene descriptions reference people (e.g., "a black background with two people")
- Only environmental details are needed (e.g., "a black background")
- VLM sees the whole image holistically and describes everything together

## Proposed Solution

### Two-Mode System

#### Mode 1: Standard (VRAM-Efficient)
For users with limited VRAM or those who prefer simpler processing:
- Uses existing VLM-only approach
- Implements scene masking to fix person contamination
- No additional model loading required
- Faster processing time

#### Mode 2: Enhanced (LLM-Assisted)
For users with sufficient VRAM and requiring maximum quality:
- Three-stage pipeline: Tags → VLM → LLM Fusion
- Adaptive enhancement based on tag coverage quality
- Scene masking for clean background descriptions
- Requires loading small LLM model (7-8B parameters)

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Fine-Tune Captioner                       │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌──────────────────┐                   │
│  │ Person Tagger  │  │ Scene Generator  │                   │
│  │ (WD Tagger)    │  │ (VLM)            │                   │
│  └────────┬───────┘  └────────┬─────────┘                   │
│           │                   │                              │
│           ↓                   ↓                              │
│  ┌────────────────────────────────────┐                     │
│  │     Mode Selection                 │                     │
│  │  ○ Standard (VRAM-Efficient)       │                     │
│  │  ○ Enhanced (LLM-Assisted)         │                     │
│  └────────┬───────────────────────────┘                     │
│           │                                                  │
│     ┌─────┴──────┐                                          │
│     │            │                                          │
│     ↓            ↓                                          │
│ ┌───────┐   ┌────────────────┐                             │
│ │Standard│   │  LLM Pipeline  │                             │
│ │Output │   │  (Enhanced)    │                             │
│ └───────┘   └────────┬───────┘                             │
│                      │                                      │
│                      ↓                                      │
│            ┌──────────────────┐                             │
│            │ Gap Analysis     │                             │
│            │ & Fusion         │                             │
│            └──────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Scene Masking System
**Purpose:** Remove people from scene description generation

**Process:**
1. Use existing YOLO person detection to get bounding boxes
2. Create masked version of image:
   - Option A: Fill person regions with median background colour
   - Option B: Apply Gaussian blur to person regions
3. Feed masked image to VLM for scene description
4. Use constrained prompt: "Describe the background setting and environment"

**Fallback:** If masking fails, add bad_words: ["people", "person", "man", "woman", "humans", "individuals"]

#### 2. Tag Coverage Analyser
**Purpose:** Determine quality of VLM description and identify gaps

**Process:**
1. Parse VLM description into semantic tokens
2. Compare against important tags (filtered for relevance)
3. Calculate coverage score: `(matched_tags / total_important_tags) * 100`
4. Generate list of missing critical details

**Tag Importance Rules:**
- Physical attributes: High priority (hair colour, eye colour, clothing)
- Generic tags: Low priority (1girl, solo, looking at viewer)
- Artistic tags: Medium priority (detailed, high quality)

#### 3. LLM Fusion Engine (Enhanced Mode Only)
**Purpose:** Enhance VLM descriptions with missing tag details

**Adaptive Processing:**
- **Coverage ≥80%:** No enhancement needed, use VLM output directly
- **Coverage 50-79%:** Light enhancement, add missing details naturally
- **Coverage <50%:** Heavy rewrite, incorporate all important tags

**LLM Prompt Template (Light Enhancement):**
```
You are enhancing training captions for AI image generation models.

ORIGINAL DESCRIPTION (from vision analysis):
{vlm_description}

MISSING SPECIFIC DETAILS (from tag analysis):
{missing_tags_only}

TASK: Rewrite the description in 2-3 sentences to naturally incorporate the missing details while preserving the narrative flow and style of the original. Be specific and detailed but maintain natural language.

Enhanced description:
```

**LLM Prompt Template (Heavy Rewrite):**
```
You are creating detailed training captions for AI image generation models.

BASIC DESCRIPTION: {vlm_description}

ALL DETECTED ATTRIBUTES: {all_important_tags}

TASK: Write a comprehensive 2-3 sentence description that naturally incorporates all the specific attributes listed above. Maintain a flowing, natural style suitable for image generation training data.

Description:
```

### Data Flow

#### Standard Mode Flow
```
1. Load directory of images
2. For each person detected:
   a. Generate WD Tagger tags
   b. Generate VLM description (no masking)
3. For scene:
   a. Apply person masking
   b. Generate VLM description on masked image
4. Combine: "person1: {desc}, person2: {desc}, scene: {desc}"
```

#### Enhanced Mode Flow
```
1. Load directory of images
2. Load LLM model (once, reuse for all images)
3. For each person detected:
   a. Generate WD Tagger tags
   b. Generate VLM description
   c. Analyse tag coverage
   d. If coverage < 80%:
      - Identify missing tags
      - Generate LLM enhancement prompt
      - Get enhanced description from LLM
4. For scene:
   a. Apply person masking
   b. Generate VLM description on masked image
5. Combine: "person1: {enhanced_desc}, person2: {enhanced_desc}, scene: {desc}"
```

## Implementation Details

### New Classes/Functions

#### `CaptionEnhancer` (New Class)
Located: `taggui/auto_captioning/utils/caption_enhancer.py`

**Responsibilities:**
- Load and manage LLM model (lazy loading)
- Perform tag coverage analysis
- Generate enhancement prompts based on coverage
- Run LLM inference for caption enhancement
- Handle model quantisation and device placement

**Key Methods:**
```python
def __init__(self, model_name: str, device: str, quantize: bool)
def load_model() -> None
def analyse_coverage(tags: List[str], description: str) -> Tuple[float, List[str]]
def enhance_description(original_desc: str, missing_tags: List[str], coverage: float) -> str
def unload_model() -> None
```

#### `ImageMasker` (New Class)
Located: `taggui/auto_captioning/utils/image_masker.py`

**Responsibilities:**
- Apply masking to person regions in images
- Support multiple masking strategies
- Validate mask quality
- Generate debug visualisations

**Key Methods:**
```python
def __init__(self, strategy: str = "median")  # "median" or "blur"
def mask_persons(image: PIL.Image, bboxes: List[Tuple[int, int, int, int]]) -> PIL.Image
def validate_mask_quality(original: PIL.Image, masked: PIL.Image) -> bool
def visualise_mask(image: PIL.Image, bboxes: List[Tuple[int, int, int, int]]) -> PIL.Image
```

### Modifications to Existing Classes

#### `MultiPersonTagger`
Located: `taggui/auto_captioning/models/multi_person.py`

**Changes:**
1. Add `caption_mode` parameter with options: "standard", "enhanced"
2. Integrate `ImageMasker` for scene generation
3. Integrate `CaptionEnhancer` for enhanced mode
4. Add coverage threshold configuration
5. Implement adaptive enhancement logic

**New Parameters:**
- `caption_mode: str` - "standard" or "enhanced"
- `llm_model_name: str` - e.g., "Qwen/Qwen2.5-7B-Instruct"
- `llm_quantize: bool` - Use 4-bit quantisation
- `enhancement_threshold: float` - Coverage threshold (default 0.8)
- `masking_strategy: str` - "median" or "blur"
- `show_debug_masks: bool` - Visualise masks in UI (development)

#### `AutoCaptioner` Widget
Located: `taggui/widgets/auto_captioner.py`

**Changes:**
1. Add caption mode radio buttons ("Standard" / "Enhanced")
2. Show/hide LLM settings based on mode
3. Add LLM model selector dropdown
4. Add quantisation checkbox
5. Add enhancement threshold slider
6. Add masking strategy selector
7. Display coverage statistics in progress updates

## Configuration

### Settings Schema

Add to `QSettings` under `auto_captioning/fine_tune/`:

```python
{
    'caption_mode': 'standard',  # 'standard' or 'enhanced'
    'llm_model_name': 'Qwen/Qwen2.5-7B-Instruct',
    'llm_quantize': True,
    'enhancement_threshold': 0.8,  # 0.0-1.0
    'masking_strategy': 'median',  # 'median' or 'blur'
    'show_debug_masks': False
}
```

### UI Layout

#### Caption Mode Section (New)
```
┌─────────────────────────────────────┐
│ Caption Enhancement                 │
├─────────────────────────────────────┤
│ ○ Standard (VRAM-Efficient)         │
│ ● Enhanced (LLM-Assisted)           │
└─────────────────────────────────────┘
```

#### LLM Settings (Visible only in Enhanced Mode)
```
┌─────────────────────────────────────┐
│ LLM Settings                        │
├─────────────────────────────────────┤
│ Model: [Qwen/Qwen2.5-7B-Instruct ▼]│
│ ☑ Use 4-bit Quantisation            │
│                                     │
│ Enhancement Threshold: [80%]        │
│ └─────────────────────────┘         │
│                                     │
│ Masking Strategy: [Median Fill ▼]  │
└─────────────────────────────────────┘
```

### Recommended LLM Models

Priority order (best quality to memory efficiency):

1. **Qwen/Qwen2.5-7B-Instruct** (Recommended)
   - Excellent instruction following
   - Good balance of quality and speed
   - ~4GB VRAM (4-bit quantised)

2. **meta-llama/Llama-3.1-8B-Instruct**
   - Strong natural language generation
   - Widely tested and reliable
   - ~5GB VRAM (4-bit quantised)

3. **microsoft/Phi-3.5-mini-instruct** (Fastest)
   - Smallest model (3.8B parameters)
   - Fastest inference
   - ~2GB VRAM (4-bit quantised)

All models support 4-bit quantisation via BitsAndBytes for reduced VRAM usage.

## Technical Specifications

### Performance Targets

**Standard Mode:**
- Processing time: ~2-3 seconds per person
- VRAM usage: Same as current (depends on VLM selected)
- No additional model loading

**Enhanced Mode:**
- Processing time: ~3-5 seconds per person (additional ~1-2s for LLM)
- VRAM usage: Current VLM + ~2-5GB for LLM (quantised)
- One-time LLM loading at start of batch

### Memory Management

**LLM Model Loading Strategy:**
1. Lazy load on first enhanced mode caption generation
2. Keep loaded during batch processing
3. Unload when switching back to standard mode
4. Unload when closing AutoCaptioner widget
5. Automatic unload if CUDA out of memory error

**VRAM Allocation Priority:**
```
High Priority: VLM (required for any mode)
Medium Priority: WD Tagger (required for fine-tune mode)
Low Priority: LLM (optional, enhanced mode only)
```

### Error Handling

**LLM Loading Failures:**
- Show error dialog: "Unable to load LLM model. Falling back to Standard mode."
- Automatically switch to Standard mode
- Log error details for troubleshooting

**CUDA Out of Memory:**
- Attempt to unload LLM and retry
- Suggest enabling quantisation if not already enabled
- Suggest switching to Standard mode
- Offer to reduce VLM batch size

**Coverage Analysis Failures:**
- Skip enhancement, use VLM output directly
- Log warning for debugging

### Quality Metrics

Track and optionally display:
- Average coverage score across batch
- Enhancement rate (% of descriptions enhanced)
- Processing time breakdown (tags/VLM/LLM)
- Memory usage per stage

## Testing Strategy

### Unit Tests

1. **Tag Coverage Analyser:**
   - Test coverage calculation with known tag/description pairs
   - Test missing tag identification
   - Test tag importance filtering

2. **Image Masking:**
   - Test median fill strategy
   - Test Gaussian blur strategy
   - Test mask validation
   - Test edge cases (no persons, overlapping boxes)

3. **LLM Enhancement:**
   - Test prompt generation for different coverage levels
   - Test model loading/unloading
   - Test quantisation options
   - Test error handling

### Integration Tests

1. **End-to-End Standard Mode:**
   - Load test images with multiple people
   - Generate captions in standard mode
   - Verify scene descriptions don't mention people
   - Check person descriptions are natural

2. **End-to-End Enhanced Mode:**
   - Generate captions with different coverage scenarios
   - Verify adaptive enhancement logic
   - Check enhanced descriptions contain tag details
   - Verify naturalness is maintained

3. **Memory Management:**
   - Test model loading/unloading cycles
   - Test switching between modes
   - Monitor VRAM usage throughout
   - Test OOM error recovery

### Manual Testing Checklist

- [ ] UI correctly shows/hides LLM settings based on mode
- [ ] LLM model dropdown populated with supported models
- [ ] Progress bar updates during LLM enhancement
- [ ] Coverage statistics displayed correctly
- [ ] Standard mode produces clean scene descriptions
- [ ] Enhanced mode produces detailed person descriptions
- [ ] Switching modes doesn't cause crashes
- [ ] Settings persist between sessions
- [ ] Debug mask visualisation works (if enabled)
- [ ] Error messages are clear and actionable

## Future Enhancements

### Potential Improvements

1. **Custom LLM Prompts:**
   - Allow users to customise enhancement prompts
   - Template system with variables
   - Save/load prompt presets

2. **Quality Feedback Loop:**
   - Allow users to rate generated captions
   - Use ratings to adjust coverage thresholds
   - Build dataset of high-quality examples

3. **Advanced Coverage Analysis:**
   - Semantic similarity scoring (not just exact matches)
   - Use CLIP embeddings to measure description-tag alignment
   - Weight tags by visual importance

4. **Multi-Stage Enhancement:**
   - First pass: Structure and composition
   - Second pass: Detail enrichment
   - Third pass: Style and polish

5. **Cloud LLM Option:**
   - Support API-based LLMs (OpenAI, Anthropic) for users without local compute
   - Configurable API keys and endpoints
   - Cost estimation and tracking

## Implementation Phases

### Phase 1: Scene Masking (Low Risk)
- Implement `ImageMasker` class
- Integrate into scene generation
- Test scene description quality
- **Estimated effort:** 1-2 days

### Phase 2: Coverage Analysis (Medium Risk)
- Implement `CaptionEnhancer.analyse_coverage()`
- Unit tests for coverage calculation
- Logging and metrics
- **Estimated effort:** 1-2 days

### Phase 3: LLM Integration (High Risk)
- Implement `CaptionEnhancer` LLM loading and inference
- Prompt engineering and testing
- Error handling and fallbacks
- **Estimated effort:** 2-3 days

### Phase 4: UI Integration (Medium Risk)
- Add mode selection to AutoCaptioner widget
- Add LLM settings controls
- Progress reporting enhancements
- **Estimated effort:** 1-2 days

### Phase 5: Testing & Polish (Low Risk)
- Integration testing
- Performance optimization
- Documentation updates
- **Estimated effort:** 2-3 days

**Total estimated effort:** 7-12 days

## Success Criteria

The implementation is successful if:

1. **Standard Mode:**
   - Scene descriptions never mention people (95%+ success rate)
   - Processing time ≤3 seconds per person
   - No additional VRAM usage

2. **Enhanced Mode:**
   - Coverage scores improve by ≥30% on average
   - Enhanced descriptions read naturally (human evaluation)
   - Critical tag details are included (≥90% of high-priority tags)
   - Processing time ≤5 seconds per person
   - VRAM usage ≤5GB additional (with quantisation)

3. **System Quality:**
   - No crashes or data loss
   - Graceful error handling and recovery
   - Clear user feedback during processing
   - Settings persist correctly

4. **User Experience:**
   - Mode selection is intuitive
   - LLM setup is straightforward
   - Progress reporting is clear
   - Results meet or exceed user expectations

---

**Document Version:** 1.0
**Date:** 2025-11-24
**Status:** Design Approved - Ready for Implementation
