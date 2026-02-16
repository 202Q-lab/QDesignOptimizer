# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-16

### Breaking Changes
- **Migrated from `qiskit-metal` to `quantum-metal`**
  - Users must recreate their virtual environment (not just update)
  - See migration guide in installation documentation
- **Updated GUI framework from PySide2 to PySide6 (6.8+)**
  - Import paths remain `qiskit_metal` for backward compatibility

### Changed
- Updated numpy compatibility for pyEPR
- Updated matplotlib and pyEPR dependencies
- Improved installation documentation

### Fixed
- Fixed pyEPR numpy compatibility issues
- Fixed dependency conflicts requiring environment recreation

## [0.0.2]

### Added
- Two mode examples: 
  - Example with a flux-tunable coupler  
  - General example demonstrating the capabilities of the `anmod` method  
- Surface participation ratio analysis  
- Upload of publication data and scripts to the repository  

### Changed
- Documentation improvements and edits  
- Refactored `design_analysis` class into `design_analysis` and `anmod_optimizer`  

### Fixed
- Prefixed Qiskit Metal components with `_` to enable surface participation ratio analysis  

## [0.0.1]

### Added
- Initial public version of the `qdesignoptimizer`