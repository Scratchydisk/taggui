# Multi-Person Tagging Roadmap

## Overview
Integrate multi-person detection and per-person tagging capabilities into TagGUI to automatically identify, tag, and organise tags for images containing multiple people.

## Phase 1: Auto-Captioner Integration (Current)
**Goal**: Add multi-person tagging as an auto-captioner model option.

**Deliverables**:
- Person detection using YOLOv8
- Per-person tag generation using existing WD Tagger
- Scene/background tag extraction
- Structured output format: `person1: tags, person2: tags, scene: tags`
- Reusable component architecture

**Benefits**:
- Works with existing TagGUI workflow
- No changes to core data model
- User can review and edit generated tags manually
- Minimal UI changes required

**Timeline**: 2-3 days

## Phase 2: Enhanced Multi-Person Mode (Future)
**Goal**: Dedicated UI for managing per-person tags with visual assignment.

**Features**:
- Bounding box visualisation overlay on images
- Separate tag lists per detected person
- Visual person identifier labels (Person 1, Person 2, etc.)
- Drag-and-drop tag assignment between people
- Per-person tag editing interface
- Person detection metadata storage

**Benefits**:
- Precise tag-to-person association
- Visual verification of detections
- Easier correction of misattributed tags
- Better dataset organisation for multi-person training data

**Challenges**:
- Requires data model changes (one tag list → multiple per image)
- New UI widgets and interactions
- Metadata file format for detection coordinates
- Backwards compatibility with existing .txt files

**Timeline**: 1-2 weeks

## Phase 3: Advanced Features (Future)
**Goal**: Face recognition and identity persistence across images.

**Features**:
- Face recognition using InsightFace/DeepFace
- Identity assignment and naming (Person A, Person B, or custom names)
- Identity persistence across multiple images
- Age and gender estimation metadata
- Identity-based filtering and search
- Identity gallery view

**Benefits**:
- Consistent tagging of the same person across dataset
- Character/subject identification for dataset organisation
- Demographics metadata for model training
- Advanced dataset curation workflows

**Dependencies**:
- Phase 2 completion
- Additional libraries: insightface or deepface
- Face embedding storage and matching system

**Timeline**: 1-2 weeks

## Technical Architecture Evolution

### Phase 1 (Current)
```
Auto-Captioner → Multi-Person Model → [Detection → Per-Person Tagging → Scene Tags] → Single .txt file
```

### Phase 2
```
Image → Detection → Per-Person UI → Individual Tag Lists → Metadata .json + .txt files
```

### Phase 3
```
Image → Detection → Face Recognition → Identity Matching → Named Per-Person Tags → Identity Database
```

## Dependencies

### Phase 1
- ultralytics (YOLOv8) - ~8MB package
- YOLOv8 model weights - ~50MB download
- Existing WD Tagger ONNX models

### Phase 2
- JSON metadata storage format
- Qt graphics view for bounding box overlay
- Extended data model

### Phase 3
- insightface or deepface
- Face embedding database (SQLite or similar)
- Identity matching algorithm

## Success Metrics

### Phase 1
- ✓ Successfully detects 90%+ of people in multi-person images
- ✓ Generates accurate per-person tags
- ✓ Processing time <3 seconds per person
- ✓ No breaking changes to existing functionality

### Phase 2
- ✓ Users can visually verify and correct person detections
- ✓ Tag reassignment workflow is intuitive
- ✓ No performance degradation with metadata
- ✓ Backwards compatible with Phase 1 outputs

### Phase 3
- ✓ Face recognition accuracy >85% for same person across images
- ✓ Identity persistence maintained across sessions
- ✓ Identity-based dataset organisation workflows functional

## Risk Mitigation

### Phase 1
- **Risk**: YOLOv8 false positives/negatives
  - *Mitigation*: Configurable confidence thresholds, minimum size filters
- **Risk**: Performance with many people
  - *Mitigation*: Max person limit, progress feedback, background processing
- **Risk**: Output format doesn't fit workflow
  - *Mitigation*: User feedback iteration, configurable format options

### Phase 2
- **Risk**: Breaking existing data model
  - *Mitigation*: Maintain .txt compatibility, metadata in separate .json files
- **Risk**: UI complexity
  - *Mitigation*: Optional feature toggle, iterative design with user testing

### Phase 3
- **Risk**: Face recognition accuracy issues
  - *Mitigation*: Multiple model options, confidence thresholds, manual override
- **Risk**: Privacy concerns with face databases
  - *Mitigation*: Local-only storage, clear documentation, optional feature
