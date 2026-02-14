# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-14

### Breaking Changes
- **Terminology Update**: Renamed "design_var" to "control_var" throughout the codebase to make terminology domain-agnostic
  - `OptTarget.design_var` → `OptTarget.control_var`
  - `OptTarget.design_var_constraint` → `OptTarget.control_var_constraint`
  - `ANModOptimizer._minimize_for_design_vars()` → `ANModOptimizer._minimize_for_control_vars()`
  - `ANModOptimizer.calculate_target_design_var()` → `ANModOptimizer.calculate_target_control_vars()`
  - `ANModOptimizer._constrain_design_value()` → `ANModOptimizer._constrain_control_value()`
  - `DesignAnalysis.optimize_target()` parameter: `updated_design_vars_input` → `updated_control_vars_input`
  - `DataExtractor.get_design_var_name_for_param()` → `DataExtractor.get_control_var_name_for_param()`
  - `DataExtractor.get_design_var_for_param()` → `DataExtractor.get_control_var_for_param()`
  - `Plotter.plot_design_vars_vs_iteration()` → `Plotter.plot_control_vars_vs_iteration()`
  - `plot_progress()` parameter: `plot_design_variables` → `plot_control_variables`
  - **Migration Guide**: Update all `OptTarget` instantiations to use `control_var` and `control_var_constraint` instead of `design_var` and `design_var_constraint`
  - **Rationale**: Make terminology domain-agnostic for use beyond circuit design applications

### Changed
- Improved error messages for boundary constraint warnings to reference "control variables"
- Updated all documentation and docstrings to use "control variable" terminology

### Note
- Naming utility functions (`design_var_lj()`, `design_var_width()`, etc.) remain unchanged as they are circuit-design-specific utilities

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