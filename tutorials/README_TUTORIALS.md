# Tutorials

Three tutorial examples are provided, ranging from a single-qubit chip to a scalability benchmark with thousands of parameters.

## Where to start

**New to QDO?** Start with `examples_coupled_transmon_chip/` and open `main_eigenmode_single_qubit_resonator.ipynb`. It covers a single transmon qubit with a readout resonator — the simplest complete workflow from chip layout to optimized device parameters. Once familiar with the basics, continue in the same folder:

- Move to `main_eigenmode_two_qubit_resonator.ipynb` to add a second qubit and a coupler.
- Use `main_capacitance_target_resonator.ipynb` or `main_kappa_target_resonator.ipynb` if you need faster simulations for specific targets.
- Use `main_charge_line_decay_single_qubit.ipynb` if T1 limits from control lines are relevant to your design.

**Working with a tunable coupler?** Go directly to `examples_coupled_transmons_tunable_coupler_chip/` after completing the single-qubit notebook.

**Interested in large-scale optimization or the ANMod algorithm?** See `example_clusters_anmod_scaliability/` — this is an advanced standalone example, not tied to a physical chip design.

---

## 1. `examples_coupled_transmon_chip/`

A superconducting chip with one or two transmon qubits, coplanar waveguide readout resonators, a fix-frequency coupler, and a charge line. This is the primary introductory example and covers the widest range of simulation types and optimization targets.

**You will learn:**
- How to define a chip layout in Qiskit Metal and connect it to HFSS
- How to set optimization targets (frequency, anharmonicity, kappa, chi, T1, capacitance)
- The difference between eigenmode and capacitance-matrix simulations and when to use each
- How to include surface material properties and compute participation ratios (p-ratio)
- How to model qubit decoherence from a nearby charge line
- How to scale up from one qubit to two qubits with a coupler

**Notebooks:**

| Notebook | System | Simulation | Targets |
|---|---|---|---|
| `main_eigenmode_single_qubit_resonator.ipynb` | 1 qubit + resonator | Eigenmode | Qubit frequency, anharmonicity; resonator frequency, kappa, chi |
| `main_eigenmode_single_qubit_resonator_pratio.ipynb` | 1 qubit + resonator | Eigenmode + surface properties | Same as above + p-ratio; includes parametric sweep of coupling width |
| `main_eigenmode_two_qubit_resonator.ipynb` | 2 qubits + resonators + coupler | Eigenmode | All of the above for both qubits + coupler frequency; uses progressive mesh refinement |
| `main_capacitance_target_resonator.ipynb` | Resonator tee junction | Capacitance matrix | Coupling capacitance between CPW lines |
| `main_kappa_target_resonator.ipynb` | Resonator + feedline | Capacitance-derived kappa | Resonator decay rate into feedline (λ/4 resonator) |
| `main_charge_line_decay_single_qubit.ipynb` | 1 qubit + charge line | Capacitance-derived T1 | Qubit T1 limit set by charge line proximity |

**Supporting files:**
- `design.py` — chip layout and component placement (transmons, resonators, coupler, charge lines, launch pads)
- `names.py` — naming conventions for modes and design variables
- `optimization_targets.py` — `OptTarget` definitions mapping geometry to quantum parameters
- `parameter_targets.py` — numerical target values (e.g. qubit 1 at 4.0 GHz, resonator 1 at 6.0 GHz, kappa = 1 MHz)
- `mini_studies.py` — HFSS mini-study configurations (passes, ports, surface properties, decay models)
- `plot_settings.py` — plot layouts for tracking optimization progress

---

## 2. `examples_coupled_transmons_tunable_coupler_chip/`

A two-qubit chip with an explicit tunable coupler component (`TunableCoupler01`). This example demonstrates how to extend the framework to architectures where the coupler is a more complicated, independently controlled quantum element.

**You will learn:**
- How to add a tunable coupler to a two-qubit system and include it in the optimization
- How to set targets for coupler frequency, anharmonicity, and coupler-qubit chi interactions
- How the optimization targets and parameter values differ for a tunable-coupler architecture (lower chi values, distinct coupler frequency)
- How to optimize the full two-qubit and coupler design in one optimization routine. 

**Notebooks:**

| Notebook | System | Simulation | Targets |
|---|---|---|---|
| `main_eigenmode_two_qubit_resonator.ipynb` | 2 transmons + 2 resonators + tunable coupler + charge lines | Eigenmode | Qubit/resonator/coupler frequencies and anharmonicities, kappas, qubit-resonator chi, coupler-qubit chi |


**Supporting files:** same structure as example 1 (`design.py`, `names.py`, `optimization_targets.py`, `parameter_targets.py`, `mini_studies.py`, `plot_settings.py`)

---

## 3. `example_clusters_anmod_scaliability/`

A scalability benchmark for the ANMod optimizer showcasing an abstract mathematical system of 1000 clusters with 3 parameters each (3000 parameters total), with tunable intra- and inter-cluster coupling. This example is implemented as Python scripts rather than notebooks.

**You will learn:**
- How the ANMod optimizer scales to thousands of parameters
- How to define a large parametric system using alpha/beta/gamma coupling matrices
- How to apply perturbative corrections for weak inter-cluster coupling
- How to track and report convergence across many parameters (relative error per parameter)
- How to visualize optimization convergence at scale (parameter evolution, design variable evolution, approximation accuracy)

**Scripts:**

| Script | Purpose |
|---|---|
| `main.py` | Runs the 3000-parameter optimization over 10 iterations; reports how many parameters converge per random seed |
| `scaled_system_definition.py` | Defines `ScaledSystem`: alpha (intra-cluster), beta (cross-parameter), gamma (inter-cluster) coupling; implements perturbative solve |
| `check_convergence.py` | Computes relative error per parameter at each iteration; prints convergence report and iteration at which all parameters converge |
| `plot_convergence.py` | Generates an 8-panel convergence figure: parameter values, design variables, y/y_target ratios, approximation accuracy histograms, and convergence slope estimate |

**Expected results:** with most random seeds, 2900-3000 of 3000 parameters converge within 10 iterations.
