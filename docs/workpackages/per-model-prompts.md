# Work Package: Per-Model Prompt Configuration

## Overview
Enable users to configure and save different prompts for each captioning model and enhancement mode, with a dedicated Prompts tab in the Settings dialog.

## Completed Work

### 1. Settings Dialog with Tabs ✅
- Added tabbed interface to Settings dialog (General, Prompts)
- General tab contains existing settings
- Prompts tab provides dedicated space for prompt editing

### 2. Unified Prompt Editor ✅
- **Prompt Location** dropdown to select what to edit:
  - Primary Captioning Model (per-VLM model prompts)
  - Enhancement: Light Mode (for coverage 50-80%)
  - Enhancement: Heavy Rewrite (for coverage <50%)
- **Model selector** shown only for Primary Captioning Model
- **Large text editor** with proper editing space
- **Template variable buttons** that insert at cursor:
  - Primary: `{tags}`, `{name}`, `{directory}`
  - Enhancement: `{original_desc}`, `{missing_tags}`, `{all_tags}`
- **Reset to Default** button for each prompt type

### 3. Per-Model VLM Prompts ✅
- Storage: `prompt_{sanitised_model_id}` keys in QSettings
- First-time use loads model's default prompt via `get_default_prompt()`
- User changes auto-save to settings
- Fixed Florence2 `get_default_prompt()` to use `@classmethod`

### 4. Enhancement Prompts from Settings ✅
- CaptionEnhancer reads light/heavy prompts from QSettings
- Settings keys: `enhancement_prompt_light`, `enhancement_prompt_heavy`
- Falls back to hardcoded defaults if not customised

### 5. Removed Runtime Prompt UI ✅
- Removed prompt text edit from Auto Captioner right panel
- Removed prompt label container and reset button
- Prompts are now configuration, not runtime settings

## Files Modified
- `taggui/dialogs/settings_dialog.py` - Added tabs and Prompts tab UI
- `taggui/widgets/auto_captioner.py` - Removed prompt UI, added `_get_prompt_for_model()`
- `taggui/auto_captioning/utils/caption_enhancer.py` - Read prompts from settings
- `taggui/auto_captioning/models/florence_2.py` - Fixed `get_default_prompt()` decorator

## Technical Details

### Settings Keys
```
prompt_{sanitised_model_id}     # Per-model VLM prompts
enhancement_prompt_light        # Light enhancement template
enhancement_prompt_heavy        # Heavy rewrite template
```

Where `sanitised_model_id` has `/` and `\` replaced with `_`.
Example: `microsoft/Florence-2-large` → `prompt_microsoft_Florence-2-large`

### Template Variables
**Primary Captioning Model:**
- `{tags}` - WD tags from the image
- `{name}` - Image filename
- `{directory}` - Image directory path

**Enhancement Prompts:**
- `{original_desc}` - VLM-generated description
- `{missing_tags}` - Tags not covered by VLM description
- `{all_tags}` - All important tags for heavy rewrite

### Default Prompts
Each model class defines `get_default_prompt()` returning its default.
Enhancement defaults are defined in both:
- `settings_dialog.py` (for UI)
- `caption_enhancer.py` (for runtime fallback)

## Future Enhancements 💡
- Export/import prompt presets
- Copy prompt from one model to another
- Indicate when prompt differs from default
- Multi-Person Tagger description prompt editing
