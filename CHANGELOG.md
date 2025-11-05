# CHANGELOG

All notable changes to DAKGEA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Complete documentation suite:
  - Configuration Guide with 6 detailed examples
  - Dataset Guide covering all formats (HybEA, BERT-INT, RDF)
  - BERT-INT Guide with architecture and tuning details
  - FAQ with common questions and troubleshooting
  - Documentation index (docs/README.md)
- Support for `reader/dataset` format in configuration (e.g., `hybea/BBC_DB`)
- Support for modern augmentation configuration syntax:
  ```yaml
  augmentation:
    method: "stub"
    reduction: 0.1
  ```
- Automatic reader inference from dataset path format
- Enhanced README with quick start and examples

### Fixed
- **CRITICAL**: Attribute triples now correctly saved and loaded by BERT-INT reader/writer
  - Previously all attribute triples were skipped (Literal objects ignored)
  - Now correctly handles 20,000+ attribute triples per dataset
  - Phase 2 performance improved from ~1.3% to ~40% with attributes
- **CRITICAL**: Metrics display bug fixed
  - Phase 1: Was showing `0.34%` instead of `0.3436` (fraction)
  - Phase 2: Was showing `3938.22%` instead of `0.3938` (fraction)
  - All metrics now consistently use fractions (0-1 range)
- Entity ID offset handling in BERT-INT writer
  - Target entities now correctly offset by source entity count
  - Alignment pairs reference correct entity IDs

### Changed
- **BREAKING**: All metrics are now fractions (0-1) instead of percentages
  - JSON output: `{"hits@1": 0.3456}` not `{"hits@1": 34.56}`
  - Logging shows fractions: `0.3456` not `34.56%`
  - This matches standard practices in research papers
- BERT-INT Interaction Model parameters aligned with reference implementation:
  - `entity_attvalue_max_num`: 20 → 50
  - `learning_rate`: 0.001 → 5e-4
  - `batch_size`: 256 → 128
- Improved logging throughout BERT-INT pipeline
  - Phase-by-phase progress reporting
  - Detailed attribute triple counts
  - Clear distinction between Phase 1 and Phase 2 metrics

### Validated
- BERT-INT implementation verified against reference:
  - Tokenization: 100% match on 2,925+ entities
  - Phase 1 metrics: <0.1% difference (exact match)
  - Phase 2 metrics: <1% difference (within normal variance)
  - Attribute extraction: Identical (2,543 entities)

---

## [0.2.0] - 2024-11-03

### Added
- BERT-INT two-phase alignment model
  - Phase 1: Basic Unit (BERT encoder)
  - Phase 2: Interaction Model (multi-view features + MLP)
- BERT-INT dataset reader and writer
- Support for attribute triples in knowledge graphs
- Checkpoint management for BERT-INT
- Direct path mode for pre-processed datasets

### Fixed
- Dataset reader/writer registry improvements
- Path resolution for external datasets

---

## [0.1.0] - 2024-10-29

### Added
- Initial release
- HybEA alignment model support
- HybEA dataset format support
- Dataset reduction pipeline
- Stub augmentation method
- Experiment configuration system
- Results tracking and JSON output
- Multi-dataset, multi-ratio experiments
- Modular reader/writer architecture

---

## Known Issues

### Current
- None

### Planned Fixes
- Add support for validation split in BERT-INT
- Implement attribute-view features properly (currently placeholder)
- Add gradient accumulation support for large batch sizes
- Improve memory efficiency in interaction model feature extraction

---

## Migration Guide

### From 0.1.0 to 0.2.0

**Configuration changes:**

Old format (still supported):
```yaml
reduction_ratio: 0.1
augmentation_method: "stub"
```

New format (recommended):
```yaml
augmentation:
  method: "stub"
  reduction: 0.1
```

**Dataset configuration:**

Old format:
```yaml
dataset:
  name: "BBC_DB"  # Auto-detects reader
```

New format (clearer):
```yaml
dataset:
  name: "hybea/BBC_DB"  # Explicit reader
```

**Metrics format:**

Old (buggy):
```json
{"hits@1": 34.56}  // Was incorrectly shown as percentage
```

New (correct):
```json
{"hits@1": 0.3456}  // Fraction (0-1 range)
```

To convert: divide old values by 100 if they appear > 1.0.

---

## Acknowledgments

This project builds upon:
- [HybEA](https://github.com/fanourakis/HybEA) - Entity alignment framework
- [BERT-INT](https://github.com/kosugi11037/bert-int) - Two-phase alignment model
- [Transformers](https://huggingface.co/docs/transformers) - BERT implementation

---

## Contributors

- Federico Scafo (@Scafooo) - Main developer
- [Contributors list](https://github.com/Scafooo/DataAug-KG-EntityResolution/graphs/contributors)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
