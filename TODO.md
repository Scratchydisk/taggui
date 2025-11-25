# TODO

## Bugs to Fix

### Enhancement Threshold Not Working
**Priority**: High
**Status**: Pending

**Issue**: Enhancement threshold set to 1.0 makes no difference - still describing full image instead of individuals + scene.

**Expected Behavior**: When enhancement threshold = 1.0, coverage will always be < 1.0, so enhancement should always be used.

**Actual Behavior**: Setting threshold to 1.0 produces no change in behavior.

**Investigation Needed**:
1. Is threshold being read correctly from settings?
2. Is coverage calculation working properly?
3. Is enhancement_mode check correct?
4. Might be calling wrong method for full image description

**Code Locations**:
- `taggui/auto_captioning/models/multi_person_tagger.py`
  - `describe_person_region()` - VLM description with optional LLM enhancement
  - `_generate_fine_tune_caption()` - Main fine-tune caption generation

**Related Settings**:
- `enhancement_threshold` (default: 0.8)
- `enhancement_mode` ('standard' or 'enhanced')
- `caption_mode` ('lora_tags' or 'fine_tune_caption')
