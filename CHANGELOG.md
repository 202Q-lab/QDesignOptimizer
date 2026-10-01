# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

### Fixed
- **GUI no longer crashes on startup** (and, in Jupyter, no longer kills the kernel) when
  calling `MetalGUI(design)` / `create_chip_base(open_gui=True)`. The bug was upstream in
  Quantum Metal: persisted Qt window state saved to `HKCU\Software\QiskitMetal\MainWindow`
  could hand `restoreState()` an inconsistent widget tree that faulted at first paint. Fixed by
  raising the `quantum-metal` floor to 0.8.1, which contains the upstream fix
  (qiskit-metal #1048 / #1122). Machines already in a poisoned state need the persisted layout
  cleared once -- see the Troubleshooting section of the installation docs.

### Changed
- `quantum-metal` requirement raised from `>=0.7.4` to `>=0.8.1, <0.9`.
- `pyside6` capped at `<6.11`. Qt 6.11 is a known suspect for a remaining GUI crash on Windows
  machines with integrated GPUs, and Quantum Metal 0.8.1 is validated against the 6.10 series.
- `pyEPR-quantum` requirement raised from `>=1.0.0` to `>=1.0.1`, matching what
  `quantum-metal` 0.8.1 requires for its Ansys integration.
- The `quantum-metal` install command in the README, installation docs and CI workflows is now
  version-pinned instead of floating to the latest release on PyPI.
- `create_chip_base` imports `MetalGUI` lazily, so `open_gui=False` no longer requires a
  working Qt stack, and passes the intended state to `toggle_docks` explicitly rather than
  relying on a stateful toggle.

### Note on progress screenshots
Quantum Metal 0.8.0 added a die outline to the matplotlib viewer (`chip_outline`, on by
default) which participates in autoscaling. The `gui.autoscale()` + `gui.screenshot()` path
used by `DesignAnalysis.screenshot` was checked on 0.8.1 and still frames to the components,
so progress screenshots are unchanged. If a future release does reframe them to the whole
die, `renderer.options.chip_outline = False` restores the component framing.


## [0.2.0] - 2026-07-08

### Added
- Support and example for partitioning the simulations of designs

### Fixed
- Two windows opening for EPR analysis
- missing _ in hardcoded capacitance names in targets

### Changed
- Updated and extended examples used in the publication https://iopscience.iop.org/article/10.1088/2058-9565/ae7ab6



## [0.1.0] - 2026-02-21

### Breaking Changes
- **Migrated from `qiskit-metal` to `quantum-metal`**
  - Users must recreate their virtual environment (not just update)
  - See migration guide in installation documentation
- **Updated GUI framework from PySide2 to PySide6 (6.10+)**
  - Import paths remain `qiskit_metal` for backward compatibility

### Changed
- Updated numpy compatibility for pyEPR
- Updated matplotlib, shapely, ipython, pandas, pyaedt and pyEPR dependencies
- Improved installation documentation

### Fixed
- Fixed pyEPR numpy compatibility issues

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
