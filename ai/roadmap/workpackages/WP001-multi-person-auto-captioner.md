# WP001: Multi-Person Auto-Captioner Model

## Objective
Implement multi-person detection and tagging as an auto-captioner model in TagGUI, enabling automatic per-person and scene tag generation for images with multiple people.

## Design Evaluation

### Approach 1: Monolithic New Model
**Description**: Create standalone `multi_person_tagger.py` with embedded YOLOv8 and WD Tagger logic.

**Pros**:
- Self-contained, all logic in one place
- No dependencies on other new modules

**Cons**:
- Duplicates existing WD Tagger ONNX code
- Not reusable for future features
- Harder to test individual components
- Violates DRY principle

**Score: 6.5/10** - Works but poor architecture

---

### Approach 2: Extend Existing WD Tagger
**Description**: Modify existing `wd_tagger.py` to add optional person detection mode.

**Pros**:
- Reuses existing code
- Single model file

**Cons**:
- Tight coupling between detection and tagging
- Complicates existing working model
- Hard to maintain two modes in one class
- Risk of breaking existing WD Tagger functionality
- Difficult to add future detection-based models

**Score: 5.5/10** - Too risky, violates single responsibility

---

### Approach 3: Composition with Utility Components
**Description**: Create reusable utility components that the multi-person model composes.

**Architecture**:
```
taggui/auto_captioning/
  utils/ (NEW)
    person_detector.py     - YOLOv8 wrapper
    scene_extractor.py     - Scene tag filtering logic
  models/
    wd_tagger.py          - Existing, no changes
    multi_person_tagger.py - Orchestrates detection + tagging
```

**Pros**:
- Clean separation of concerns
- Reusable components for future features (Phase 2, 3)
- Existing WD Tagger unchanged (zero regression risk)
- Each component independently testable
- Easy to swap detection algorithms later
- Follows composition over inheritance

**Cons**:
- More files to manage
- Need to handle component initialization

**Score: 8.5/10** - Good architecture, slight complexity

---

### Approach 3.1: Refined Composition (SELECTED)
**Description**: Enhanced version of Approach 3 with better error handling and fallbacks.

**Architecture**:
```
taggui/auto_captioning/
  utils/ (NEW)
    __init__.py
    person_detector.py     - YOLOv8 wrapper with auto-download
    scene_extractor.py     - Configurable scene tag extraction
  models/
    wd_tagger.py          - Existing WDTagger class (import and reuse)
    multi_person_tagger.py - Main orchestrator
```

**Component Design**:

**PersonDetector**:
- Encapsulates YOLOv8 model loading and inference
- Configurable: confidence threshold, min detection size, max people
- Auto-downloads model weights on first use
- Returns sorted bounding boxes (by area, top-to-bottom)
- Handles GPU/CPU device selection

**SceneExtractor**:
- Takes WD Tagger results for full image
- Filters tags using configurable scene keywords
- Returns scene-specific tags only
- Extensible keyword list

**MultiPersonTagger**:
- Inherits from `AutoCaptioningModel`
- Orchestrates: detect → crop with padding → tag each person → extract scene
- Fallback: if no detections, tags full image as standard WD Tagger
- Formats output with clear structure
- Handles errors gracefully (logs, continues)

**Output Format**:
```
person1: brown_hair, blue_shirt, sitting, smiling
person2: blonde_hair, red_dress, standing
scene: indoors, library, bookshelf, wooden_floor
```

**Error Handling**:
- No YOLO model → Auto-download with progress
- No people detected → Fall back to full-image tagging (like standard WD Tagger)
- Detection fails → Log error, skip image, continue batch
- Too many people → Process first N (configurable max), log warning
- WD Tagger fails on crop → Log error, omit that person, continue

**Performance Optimizations**:
- Batch WD Tagger inference where possible
- Reuse loaded models across images
- GPU memory management
- Progress callbacks for UI feedback

**Pros**:
- All benefits of Approach 3
- Robust error handling and fallbacks
- No user-facing failures for edge cases
- Performance considerations built-in
- Clear, maintainable code structure
- Easy to extend for Phase 2 features

**Cons**:
- Initial implementation effort slightly higher
- Need thorough testing of error paths

**Score: 9.5/10** - Production-ready architecture

**Why 9.5 not 10**: Slight complexity with multiple components, but this is justified by maintainability and extensibility.

---

## Selected Approach: 3.1 Refined Composition

## Critical Design Decision: WD Tagger Integration

### Reuse Existing WdTaggerModel Class

**Decision**: Import and reuse the existing `WdTaggerModel` class from `taggui/auto_captioning/models/wd_tagger.py` rather than reimplementing WD Tagger logic.

**Rationale**:

1. **Consistent Tag Formatting**:
   - Existing WD Tagger replaces underscores with spaces: `brown_hair` → `brown hair`
   - Exception: Kaomojis (emoticons) preserve underscores: `^_^`, `o_o`, `>_<`
   - Reusing the class ensures our multi-person tags have identical formatting

2. **Zero Code Duplication**:
   - WD Tagger ONNX inference logic already implemented and tested
   - Tag preprocessing (underscore replacement) already handled
   - CSV parsing and tag categorisation already working

3. **Automatic Consistency**:
   - Any future changes to WD Tagger logic automatically apply to multi-person tagger
   - No risk of divergent behaviour between models
   - Maintains single source of truth for WD Tagger behaviour

4. **Reduced Testing Burden**:
   - No need to test WD Tagger inference separately
   - Only need to test detection + cropping + orchestration logic

**Implementation**:
```python
from auto_captioning.models.wd_tagger import WdTaggerModel

class MultiPersonTagger(AutoCaptioningModel):
    def load_model(self):
        # PersonDetector for YOLO
        self.person_detector = PersonDetector(...)

        # Reuse existing WdTaggerModel for tagging
        self.wd_model = WdTaggerModel(self.model_id)

        # SceneExtractor for filtering
        self.scene_extractor = SceneExtractor()

    def generate_caption(self, model_inputs, image_prompt):
        # Use self.wd_model.generate_tags() for each person crop
        tags, probs = self.wd_model.generate_tags(cropped_image_array, settings)
        # Tags already have underscores replaced with spaces
```

**Key Points**:
- ✅ `WdTaggerModel` handles all tag formatting (including underscore → space)
- ✅ Kaomoji list (`^_^`, `o_o`, etc.) automatically preserved
- ✅ Rating tags excluded automatically
- ✅ Tag categories (general, character, rating) handled correctly
- ✅ Zero risk of inconsistent tag formatting across models

**Alternative Rejected**: Implementing our own ONNX inference
- ❌ Would require duplicating underscore replacement logic
- ❌ Would need to maintain Kaomoji exception list separately
- ❌ Risk of inconsistency if WD Tagger logic changes
- ❌ More code to test and maintain

**Score Impact**: This decision raises our approach from 9.5/10 to **9.8/10** by eliminating potential consistency issues.

---

## Detailed Implementation Plan

### 1. PersonDetector Utility (`taggui/auto_captioning/utils/person_detector.py`)

**Responsibilities**:
- Load and manage YOLOv8 model
- Detect people in images
- Filter and sort detections

**Class Interface**:
```python
class PersonDetector:
    def __init__(self, model_size='m', device=None,
                 conf_threshold=0.5, min_size=50, max_people=10)

    def detect_people(self, image_path: str) -> List[Dict]
        # Returns: [{'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]

    def crop_person(self, image: Image, bbox: List[int],
                    padding: int = 10) -> Image
```

**Key Features**:
- Auto-downloads YOLOv8 weights via ultralytics API
- Device auto-detection (CUDA if available)
- Filters detections by confidence and minimum size
- Sorts by bounding box area (largest first) then Y-position
- Limits to max_people detections

**Dependencies**: `ultralytics`, `torch`, `PIL`

**Error Handling**:
- Model download failure → Raise clear error with download URL
- Invalid image → Return empty list
- CUDA OOM → Fall back to CPU automatically

---

### 2. SceneExtractor Utility (`taggui/auto_captioning/utils/scene_extractor.py`)

**Responsibilities**:
- Extract scene/environment tags from full-image WD Tagger results
- Filter based on scene-related keywords

**Class Interface**:
```python
class SceneExtractor:
    def __init__(self, scene_keywords: List[str] = None)

    def extract_scene_tags(self, all_tags: List[Tuple[str, float]],
                          max_tags: int = 20) -> List[str]
```

**Default Scene Keywords**:
- Locations: indoors, outdoors, building, room, street, etc.
- Environments: forest, beach, mountain, city, park, etc.
- Settings: classroom, bedroom, kitchen, office, library, etc.
- Time/weather: night, day, sunset, cloudy, rain, snow, etc.
- Objects: furniture, table, chair, bed, bookshelf, etc.
- Background: wall, window, door, floor, ceiling, etc.

**Key Features**:
- Configurable keyword list (can be extended by users)
- Returns tags sorted by confidence
- Limits to top N scene tags (default 20)
- Handles multi-word tag matching (e.g., "wooden floor" matches keyword "floor")
  - Note: Tags received already have underscores replaced with spaces by `WdTaggerModel`

**Dependencies**: None (pure Python)

**Important**: Scene keywords should use spaces, not underscores (e.g., "wooden floor" not "wooden_floor"), since tags from `WdTaggerModel` already have underscores replaced.

---

### 3. MultiPersonTagger Model (`taggui/auto_captioning/models/multi_person_tagger.py`)

**Responsibilities**:
- Main orchestrator for multi-person tagging workflow
- Inherits from `AutoCaptioningModel`
- Manages component lifecycle

**Class Structure**:
```python
from auto_captioning.models.wd_tagger import WdTaggerModel
from auto_captioning.utils.person_detector import PersonDetector
from auto_captioning.utils.scene_extractor import SceneExtractor

class MultiPersonTagger(AutoCaptioningModel):
    # Model metadata
    name = "Multi-Person Tagger (WD + YOLO)"
    model_id = "multi-person-wd-yolo"

    def __init__(self, captioning_thread, caption_settings):
        super().__init__(captioning_thread, caption_settings)
        self.person_detector = None
        self.wd_model = None
        self.scene_extractor = None

    def load_model(self):
        # Initialize PersonDetector for YOLO
        self.person_detector = PersonDetector(
            model_size=self.caption_settings['yolo_model_size'],
            conf_threshold=self.caption_settings['detection_confidence'],
            min_size=self.caption_settings['detection_min_size'],
            max_people=self.caption_settings['detection_max_people']
        )

        # Reuse existing WdTaggerModel (handles underscore replacement)
        wd_model_id = self.caption_settings['wd_model']
        self.wd_model = WdTaggerModel(wd_model_id)

        # Initialize SceneExtractor
        self.scene_extractor = SceneExtractor()

    def generate_caption(self, image: Image, prompt: str) -> str
        # Main processing pipeline (see below)

    def _format_output(self, person_tags_list, scene_tags) -> str
        # Format structured output with person1:, person2:, scene: prefixes
```

**Processing Pipeline**:
1. Detect people using PersonDetector (YOLOv8)
2. If no people detected → Fall back to standard WD Tagger on full image
3. For each detected person:
   - Crop with padding using `PersonDetector.crop_person()`
   - Convert crop to numpy array (same preprocessing as WD Tagger)
   - Run `WdTaggerModel.generate_tags()` (returns tags with underscores → spaces)
   - Collect top N tags (without probabilities for cleaner output)
4. Run `WdTaggerModel.generate_tags()` on full image
5. Extract scene tags using `SceneExtractor.extract_scene_tags()`
6. Format output: `person1: tags, person2: tags, scene: tags`
7. Return formatted caption

**Key Implementation Detail**:
- All WD Tagger calls use `WdTaggerModel.generate_tags()`, ensuring consistent tag formatting
- No manual underscore replacement needed - handled automatically by `WdTaggerModel`
- Kaomoji exceptions (`^_^`, `o_o`, etc.) preserved automatically

**Settings Integration**:
```python
caption_settings = {
    'model_id': 'multi-person-wd-yolo',
    'detection_confidence': 0.5,      # NEW
    'detection_min_size': 50,         # NEW
    'detection_max_people': 10,       # NEW
    'crop_padding': 10,               # NEW
    'yolo_model_size': 'm',           # NEW (n/s/m/l/x)
    'include_scene_tags': True,       # NEW
    'max_scene_tags': 20,             # NEW
    'tag_threshold': 0.35,            # Existing WD Tagger setting
    'wd_model': 'SmilingWolf/wd-eva02-large-tagger-v3',  # NEW
    ... (other existing caption settings)
}
```

**Output Format Examples**:

*Single person detected*:
```
person1: brown hair, blue shirt, sitting, smiling, glasses
scene: indoors, library, bookshelf, window, natural lighting
```

*Multiple people*:
```
person1: brown hair, blue shirt, sitting, glasses, book
person2: blonde hair, red dress, standing, smiling
person3: black hair, green sweater, sitting
scene: indoors, library, bookshelf, wooden floor, warm lighting
```

*No people detected (fallback)*:
```
library, bookshelf, indoors, wooden floor, window, books, empty room, reading space
```

**Note**: All tags have underscores replaced with spaces (e.g., WD model outputs `brown_hair` but displays as `brown hair`) via `WdTaggerModel` preprocessing. Kaomojis like `^_^` preserve underscores.

**Error Handling**:
- Detection fails → Log warning, fall back to standard tagging
- Person crop fails → Log warning, skip that person
- WD Tagger fails → Log error, return empty tags for that person
- Scene extraction fails → Omit scene section
- Any component failure → Continue processing, never crash

---

### 4. Model Registration (`taggui/auto_captioning/models_list.py`)

**Changes Required**:
Add MultiPersonTagger to the model list:

```python
from auto_captioning.models.multi_person_tagger import MultiPersonTagger

# Add to appropriate section
MULTI_PERSON_MODELS = [
    {
        'name': 'Multi-Person WD Tagger (YOLOv8)',
        'model_id': 'multi-person-wd-yolo',
        'class': MultiPersonTagger,
        'description': 'Detects multiple people and tags each individually with scene context',
    }
]
```

---

### 5. Dependencies Update (`requirements.txt`)

**Additions**:
```txt
# Multi-person tagging
ultralytics>=8.0.0
```

**Notes**:
- `torch` already in requirements (PyTorch dependency)
- `PIL` already in requirements (Pillow)
- `onnxruntime` already in requirements (for WD Tagger)
- ultralytics will auto-download YOLOv8 weights (~50MB) on first use

---

## Implementation Steps

### Step 1: Create Utilities (Day 1 - Morning)
**Duration**: 3-4 hours

1. Create `taggui/auto_captioning/utils/__init__.py`
2. Implement `person_detector.py`:
   - PersonDetector class with YOLO integration
   - Test with sample images from ../tagging/images
   - Verify GPU/CPU handling
   - Test auto-download mechanism

3. Implement `scene_extractor.py`:
   - SceneExtractor class with keyword filtering
   - Test with various WD Tagger outputs
   - Verify keyword matching logic

**Testing**:
- Unit tests for PersonDetector with mock images
- Unit tests for SceneExtractor with sample tag lists
- Integration test: detect + crop pipeline

**Acceptance Criteria**:
- ✓ PersonDetector successfully detects people in test images
- ✓ Bounding boxes correctly sorted and filtered
- ✓ SceneExtractor filters scene tags accurately
- ✓ All utility functions handle errors gracefully

---

### Step 2: Implement MultiPersonTagger (Day 1 - Afternoon)
**Duration**: 4-5 hours

1. Create `multi_person_tagger.py`
2. Inherit from AutoCaptioningModel
3. Implement component initialization
4. Implement main processing pipeline
5. Implement output formatting
6. Add error handling and fallbacks

**Testing**:
- Test with single-person images
- Test with multi-person images
- Test with no-person images (fallback)
- Test error handling (corrupt images, missing models)

**Acceptance Criteria**:
- ✓ Model appears in Auto-Captioner dropdown
- ✓ Successfully processes single-person images
- ✓ Successfully processes multi-person images
- ✓ Falls back gracefully when no people detected
- ✓ Output format is correct and readable
- ✓ No crashes or unhandled exceptions

---

### Step 3: Settings Integration (Day 2 - Morning)
**Duration**: 2-3 hours

1. Add new settings to caption_settings dict
2. Wire up settings to PersonDetector initialization
3. Add settings to Settings dialog UI (if needed)
4. Test settings persistence
5. Document default values

**Testing**:
- Test different confidence thresholds
- Test different YOLO model sizes
- Test max_people limiting
- Test scene tag inclusion toggle
- Verify settings persist across sessions

**Acceptance Criteria**:
- ✓ All settings configurable via UI
- ✓ Settings persist correctly
- ✓ Default values are sensible
- ✓ Settings changes take effect immediately

---

### Step 4: Integration Testing (Day 2 - Afternoon)
**Duration**: 3-4 hours

1. Test with real user workflow:
   - Load directory with multi-person images
   - Select images
   - Configure multi-person model
   - Run captioning
   - Verify outputs

2. Test edge cases:
   - Very large images
   - Many people (>10)
   - Partially visible people
   - Occluded people
   - Low-quality images

3. Performance testing:
   - Measure processing time per image
   - Test batch processing
   - Monitor GPU memory usage
   - Verify no memory leaks

4. Compatibility testing:
   - Test alongside existing models
   - Test with different WD Tagger models
   - Test on CPU-only systems

**Acceptance Criteria**:
- ✓ Processes 1-5 people in <3 seconds per image
- ✓ No memory leaks during batch processing
- ✓ Works correctly on CPU and GPU
- ✓ Compatible with all existing functionality
- ✓ No regression in existing auto-captioner models

---

### Step 5: Documentation and Polish (Day 3)
**Duration**: 2-3 hours

1. Update CLAUDE.md with new model information
2. Add inline code documentation
3. Add example outputs to documentation
4. Create user guide section
5. Update model list with descriptions

**Documentation Sections**:
- How to use multi-person tagger
- Settings explanation
- Output format description
- Troubleshooting common issues
- Performance tips

**Acceptance Criteria**:
- ✓ Code is well-documented
- ✓ User documentation is clear
- ✓ Examples are provided
- ✓ CLAUDE.md is updated

---

## Testing Strategy

### Unit Tests
**Utilities**:
- `test_person_detector.py`:
  - Test detection with known images
  - Test confidence filtering
  - Test size filtering
  - Test max people limiting
  - Test sorting logic
  - Test cropping with padding

- `test_scene_extractor.py`:
  - Test keyword matching
  - Test tag filtering
  - Test max tags limiting
  - Test custom keyword lists

**Models**:
- `test_multi_person_tagger.py`:
  - Test initialization
  - Test processing pipeline
  - Test output formatting
  - Test fallback behaviour
  - Test error handling

### Integration Tests
- End-to-end workflow test
- Compatibility with existing models test
- Settings persistence test
- Performance benchmark test

### Manual Testing Checklist
- [ ] Install dependencies successfully
- [ ] Model appears in dropdown
- [ ] Process single-person image
- [ ] Process multi-person image (2-5 people)
- [ ] Process image with no people (fallback)
- [ ] Process batch of mixed images
- [ ] Adjust confidence threshold
- [ ] Toggle scene tags on/off
- [ ] Try different YOLO model sizes
- [ ] Verify output format is correct
- [ ] Edit generated tags manually
- [ ] Save and reload directory
- [ ] Test on CPU-only system
- [ ] Test with large images (>4K)
- [ ] Test with many people (>10)

---

## Integration Points

### Files to Modify
1. **requirements.txt** - Add ultralytics
2. **taggui/auto_captioning/models_list.py** - Register new model

### Files to Create
1. **taggui/auto_captioning/utils/__init__.py** - Package init
2. **taggui/auto_captioning/utils/person_detector.py** - YOLOv8 wrapper
3. **taggui/auto_captioning/utils/scene_extractor.py** - Scene tag logic
4. **taggui/auto_captioning/models/multi_person_tagger.py** - Main model

### Files to Read/Import
1. **taggui/auto_captioning/models/wd_tagger.py** - Import `WdTaggerModel` class
   - **Critical**: Reuse this class to ensure consistent tag formatting
   - Handles underscore → space replacement automatically
   - Preserves Kaomoji exceptions (`^_^`, `o_o`, etc.)
   - NO modification to this file required
2. **taggui/auto_captioning/auto_captioning_model.py** - Import `AutoCaptioningModel` base class
3. **taggui/utils/image.py** - Import `Image` class

### No Changes Required
- **taggui/auto_captioning/models/wd_tagger.py** - Zero modifications (import only)
- Core data model (Image, ImageListModel)
- UI components (AutoCaptioner widget)
- Settings system (works with existing)
- File I/O (.txt files unchanged)

---

## Risk Mitigation

### Risk 1: YOLOv8 Performance
**Impact**: High | **Likelihood**: Medium

**Risk**: YOLO detection too slow for real-time use.

**Mitigation**:
- Use YOLOv8m (medium) as default (balanced speed/accuracy)
- Provide smaller models (n, s) for faster processing
- Add progress feedback in UI
- Process in background thread (existing system)

---

### Risk 2: Detection Accuracy
**Impact**: Medium | **Likelihood**: Medium

**Risk**: False positives/negatives in person detection.

**Mitigation**:
- Configurable confidence thresholds
- Minimum size filtering (avoid tiny detections)
- User can manually review and edit outputs
- Fallback to standard tagging if no detections

---

### Risk 3: Output Format Usability
**Impact**: Medium | **Likelihood**: Low

**Risk**: Structured output format doesn't fit user workflows.

**Mitigation**:
- Keep format simple and editable
- Use clear, human-readable structure
- Allow manual editing post-generation
- Gather user feedback for iteration

---

### Risk 4: Memory Usage
**Impact**: Medium | **Likelihood**: Low

**Risk**: Multiple model loads consume too much GPU memory.

**Mitigation**:
- Share WD Tagger model instance across crops
- Proper cleanup after processing
- Support CPU fallback
- Monitor memory usage during testing

---

### Risk 5: Dependency Conflicts
**Impact**: High | **Likelihood**: Low

**Risk**: ultralytics conflicts with existing dependencies.

**Mitigation**:
- Test in clean venv before integration
- Check ultralytics compatibility with torch versions
- Document any version constraints
- Provide clear error messages if conflicts occur

---

## Success Criteria

### Functional Requirements
- ✓ Detects 90%+ of clearly visible people in test images
- ✓ Generates relevant tags for each detected person
- ✓ Extracts accurate scene tags
- ✓ Output format is clear and editable
- ✓ Falls back gracefully when no people detected
- ✓ Handles errors without crashing

### Performance Requirements
- ✓ Processing time <3 seconds per person on GPU
- ✓ Processing time <10 seconds per person on CPU
- ✓ Batch processing without memory leaks
- ✓ No UI freezing during processing

### Integration Requirements
- ✓ Works with existing auto-captioner workflow
- ✓ No changes to core data model
- ✓ No regression in existing models
- ✓ Compatible with all existing settings

### Code Quality Requirements
- ✓ Clean separation of concerns
- ✓ Reusable components
- ✓ Comprehensive error handling
- ✓ Well-documented code
- ✓ Unit tests for utilities
- ✓ Integration tests for model

---

## Deliverables

1. **Code**:
   - `taggui/auto_captioning/utils/person_detector.py`
   - `taggui/auto_captioning/utils/scene_extractor.py`
   - `taggui/auto_captioning/models/multi_person_tagger.py`
   - Updated `requirements.txt`
   - Updated `models_list.py`

2. **Documentation**:
   - Updated CLAUDE.md
   - Inline code documentation
   - User guide section

3. **Tests**:
   - Unit tests for utilities
   - Integration tests for model
   - Manual testing checklist completed

4. **Examples**:
   - Sample outputs documented
   - Example settings configurations

---

## Timeline

**Total Estimate**: 2-3 days (16-24 hours)

**Day 1** (8 hours):
- Morning: Utilities implementation (4h)
- Afternoon: MultiPersonTagger implementation (4h)

**Day 2** (8 hours):
- Morning: Settings integration (3h)
- Afternoon: Integration testing (5h)

**Day 3** (4-8 hours):
- Documentation and polish (3h)
- Buffer for issues/refinement (1-5h)

---

## Future Enhancements (Phase 2 Preparation)

This implementation is designed to support Phase 2 features:

**Reusable Components**:
- PersonDetector can provide bbox coordinates for UI overlay
- Scene extraction logic can be used in dedicated scene tagger
- Multi-person output format can be parsed for per-person UI

**Extension Points**:
- PersonDetector can be extended with face detection
- SceneExtractor keywords can be made user-configurable
- Output formatter can support multiple formats (JSON, structured tags)

**Metadata Foundation**:
- Detection bboxes available for future metadata storage
- Person ordering logic supports identity assignment
- Tag structure supports per-person data model

---

## Final Design Score: 9.8/10

**Strengths**:
- Clean, maintainable architecture with clear separation of concerns
- Reusable components for future phases (Phase 2, 3)
- Robust error handling with graceful fallbacks
- **Zero breaking changes** to existing code
- **100% consistency** with existing WD Tagger behaviour (via WdTaggerModel reuse)
- Performance-conscious design (GPU memory management, batch processing)
- Comprehensive testing strategy
- Automatic tag formatting (underscore → space, Kaomoji preservation)

**Minor Weaknesses**:
- Slightly more complex with multiple components (justified by benefits)
- Requires thorough testing of error paths
- Initial implementation effort higher than monolithic approach

**Justification for 9.8**: This design balances immediate functionality with long-term extensibility. The composition pattern ensures maintainability, the error handling ensures robustness, and the architecture directly supports Phase 2 and 3 enhancements. The critical decision to reuse `WdTaggerModel` eliminates consistency risks and raises the score from 9.5 to 9.8. The 0.2 point deduction is for the inherent complexity of multi-component systems, which is unavoidable (and justified) for a production-quality implementation.
