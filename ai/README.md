# TagGUI AI Development

This directory contains AI-related development plans, roadmaps, and workpackages for TagGUI.

## Structure

```
ai/
├── README.md (this file)
└── roadmap/
    ├── multi-person-tagging-roadmap.md  - Overall vision and phases
    └── workpackages/
        └── WP001-multi-person-auto-captioner.md  - Detailed implementation plan
```

## Current Focus

**Phase 1: Multi-Person Auto-Captioner Integration**

We are implementing multi-person detection and tagging as an auto-captioner model. This allows TagGUI to automatically:
- Detect multiple people in images using YOLOv8
- Generate per-person tags using WD Tagger
- Extract scene/background tags
- Output structured tags: `person1: tags, person2: tags, scene: tags`

**Status**: Planning complete, ready for implementation

## Documents

### [Multi-Person Tagging Roadmap](roadmap/multi-person-tagging-roadmap.md)
High-level vision covering three phases:
- **Phase 1**: Auto-captioner integration (current)
- **Phase 2**: Dedicated multi-person UI with bounding boxes
- **Phase 3**: Face recognition and identity persistence

### [WP001: Multi-Person Auto-Captioner](roadmap/workpackages/WP001-multi-person-auto-captioner.md)
Detailed workpackage for Phase 1 implementation:
- Design evaluation (scored 9.5/10)
- Component architecture
- Implementation steps
- Testing strategy
- Timeline: 2-3 days

## Quick Links

**Related Projects**:
- `../tagging/` - Reference implementation scripts for multi-person detection

**Key Files to Modify**:
- `requirements.txt` - Add ultralytics dependency
- `taggui/auto_captioning/models_list.py` - Register new model

**New Files to Create**:
- `taggui/auto_captioning/utils/person_detector.py`
- `taggui/auto_captioning/utils/scene_extractor.py`
- `taggui/auto_captioning/models/multi_person_tagger.py`

## Implementation Approach

**Selected Design**: Composition with Reusable Components (Score: 9.5/10)

**Key Principles**:
- ✓ No changes to existing working code
- ✓ Reusable components for future phases
- ✓ Clean separation of concerns
- ✓ Robust error handling
- ✓ Performance-conscious

**Architecture**:
```
PersonDetector (YOLOv8) + SceneExtractor (Filters) + WDTagger (Existing)
                                ↓
                    MultiPersonTagger (Orchestrator)
                                ↓
          Structured Output: person1: tags, person2: tags, scene: tags
```

## Next Steps

1. Review WP001 workpackage
2. Create development branch
3. Follow implementation steps in WP001
4. Test with sample images from `../tagging/images/`
5. Integration testing with TagGUI workflow
6. Documentation and polish

## Success Metrics

**Phase 1 Completion**:
- ✓ Successfully detects 90%+ of people in multi-person images
- ✓ Generates accurate per-person tags
- ✓ Processing time <3 seconds per person
- ✓ No breaking changes to existing functionality
- ✓ Robust error handling
- ✓ Production-ready code quality

## Questions or Issues?

Refer to:
- WP001 for detailed implementation guidance
- Roadmap for long-term vision
- `../tagging/` scripts for reference implementations
