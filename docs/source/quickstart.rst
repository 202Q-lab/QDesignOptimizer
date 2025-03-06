QDO quickstart
==============
This quickstart takes the example of a single transmon-resonator system to examplify the setup of the optimization project. Based on this concept more complex optimization problems can be constructed. This minimal example can be found in projects/example_single_transmon.


Installation
------------
For the installation we refer back to the installation guide :ref:`installation`.


Project structure
-----------------
Every optimization project requires a set of files:

.. code-block::

    project_root/
    ├── design_variables.json
    ├── design_variable_names.py
    ├── design_constants.py
    ├── design.py
    ├── main.ipynb
    ├── mini_studies.py
    ├── optimization_targets.py
    ├── plot_settings.py
    ├── parameter_targets.py

Hereafter, we will refer to each file individually and highlight the important concepts to setup the project.

Branches
--------
The concept of branches is important for the efficient optimization of larger superconducting quantum circuits to break up the optimization problem into subsets of circuit designs, which can be first evaluated independently and then in conjunction with neighboring branches. In our definition a branch can be any set of circuit elements with corresponding eigenmode. Hence even a single element can be a branch. We typically like to think of a branch as a set of a resonator, a qubit, and a coupler, or instead a set of a resonator, a qubit, and a cavity with an additional branch for the coupler.

For the component names we suggest a nameing convention of the form ``NAME_RES`` or ``NAME_QUB`` followed by the branch index. A collection of common component names can be directly called from ``utils.utils_design_variables``. User specific component names can be added in and called from ``design_variable_names.py`` in the project folder.


Design Variable Names and Values
--------------------------------
Design variable names are string identifiers (for geometric lengths, Josephson junction inductances etc.) specified in the qiskit-metal design, which are varied during the optimization. For the design variable names we suggest a naming convention of the form ``design_var_res_length_`` followed by the branch index. A collection of common design variable names can be directly called from ``utils.utils_design_variable_names.py``. User specific component names can be added in and called from ``design_variable_names.py`` in the project folder.

To render the qiskit-metal design, the user have to provide initial (sensible guess) values for all design variables, which by convention is written in ``design_variables.json`` in the project folder and provided to the optimizer.


Design
------
The user specific qiskit-metal design can be created as usual following the qiskit-metal guide line using general or custom-made circuit components. Any component arguments which should be optimized must be provided as a design variable identifier.

As a minimal example we can look at the definition of a ``RouteMeander`` resonator. In this example, we show that the design variable substitues the otherwise commonly static design definition for the total resonator length.

.. code-block:: python

    import design_variable_names as u
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

    resonator_options = dict(
        total_length=u.design_var_res_length(branch),
        trace_width='20um',
        trace_gap='20um',
    )

    RouteMeander(design, u.name_res(branch), options=resonator_options)

The user can proceed in a similar manner with all other components. For bigger projects involving multiple sets or tiles of identical components types, we suggest to wrap all components of one set into a group as a function of the form:

.. code-block:: python

    def add_branch(design: DesignPlanar, branch: int, gui: MetalGUI):
        make_transmon_plus_resonator(design=design, branch=branch)

Finally, the design can be instantiated by the ``create_chip_base`` method and rendered with the components and the design variables. A wrapper function (by convention called ``render_qiskit_metal_design``), must be created such that it can be passed into the optimizer. A minimal example looks like this:

.. code-block:: python

    import design as d
    from qdesignoptimizer.utils.utils_design import create_chip_base

    CHIP_NAME = "transmon_chip"
    OPEN_GUI = True
    CHIP_TYPE = {"size_x": "10mm",
                "size_y": "10mm",
                "size_z": "-300um"}
    design, gui = create_chip_base(chip_name=CHIP_NAME, chip_type=CHIP_TYPE, open_gui=OPEN_GUI)

    u.add_design_variables_to_design(design, dv)

    def render_qiskit_metal_design(design, gui):
        d.add_branch(design, 0, gui)

    render_qiskit_metal_design(design, gui)
    # This line will render the qiskit design in the gui, which is useful when developing the design.


Optimization Targets
--------------------
The optimization target ``OptTarget`` is the first required core component of the qdesignoptimizer. The full class documentation is to be found in src/qdesignoptimizer/design_analysis_types.py.
One ``OptTarget`` should be created for each parameter the user wants to optimize for. The core role of the ``OptTarget`` is to define the physical proportionality relation between the target parameter and all design variables as well as other parameters.
The names of the involved eigenmodes and parameter names is by convention called from ``design_constants`` in the project folder. We suggest that these target parameter names take the form ``res_freq``.

A minimal example for the resonator length can look like this:

.. code-block:: python

    from qdesignoptimizer.design_analysis_types import OptTarget
    import design_constants as dc
    import design_variable_names as u
    def get_opt_target_res_freq_via_length(
    resonator: Mode,
    design_var_res_length: Callable = n.design_var_length,
        ) -> OptTarget:

    return OptTarget(
        target_param_type=c.FREQ,
        involved_modes=[resonator],
        design_var=design_var_res_length(resonator),
        design_var_constraint={"larger_than": "500um", "smaller_than": "15000um"},
        prop_to=lambda p, v: 1 / v[design_var_res_length(resonator)],
        independent_target=True,
    )


.. caution:: Ensure that the units of the design variable matches the unit of the contrain in the optimization target and the parameters in the propotionality statement prop_to. For consistency we suggest to use the units :math:`um` for measures of length, :math:`nH` for inductances and :math:`fF` for capacitances.

One strength of the qdesignoptimizer is how it handles the physical relations between the design variable and the parameter targets, which boosts the efficiency of the optimization. Note that the ``OptTarget`` only requires an expression which is proportional to the target quantity, since it only uses relative values in the update step. Hence, the user only need to provide the part of the function which vaies and to the level of detail which is known to the user. The more accurate the user specified model is, the faster and more robust the optimizer will be. The table below contains an example set of suggested physical relations for the optimization targets for Hamiltonian and dissipative parameters in a dispersively coupled qubit-resonator cQED system.:

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 20 15

   * - **Quantity**
     - **Symbol**
     - **Proportional to**
     - **Design variable**
     - **Independence**
   * - Resonator frequency
     - :math:`f_{res}`
     - :math:`1 / l_{res}`
     - :math:`l_{res}`
     - True
   * - Qubit frequency
     - :math:`f_{qb}`
     - :math:`1 / \sqrt{L_{J,qb} \cdot w_{qb}}`
     - :math:`L_{qb}, w_{qb}`
     - False
   * - Anharmonicity
     - :math:`\alpha`
     - :math:`1 / w_{qb}`
     - :math:`w_{qb}`
     - True
   * - Dispersive shift
     - :math:`\chi`
     - :math:`w_{res-qb} \cdot \alpha / (f_{qb}-f_{res}-\alpha)`
     - :math:`w_{res-qb}`
     - False
   * - Resonator decay rate
     - :math:`\kappa_{res}`
     - :math:`l_{res-tl}`
     - :math:`l_{res-tl}`
     - True

.. caution::  An OptTarget can be marked as independent_target=True if the target only depends on a single design variable and not on any system parameter. This allows the optimizer to solve this OptTarget independently, making it faster and more robust. If a criteria of independence is not fulfilled, the OptTarget MUST be independent_target=False (as the default).

Parameter Targets
-----------------
The parameter targets are specified in a ``dict`` per target parameter. The target parameters can be called from ``design_constants``. A minimal example for a single qubit-resonator system may look like this:
# TODO AXEL  Decription: Mode, nonlinearity, capacitance matrix
# TODO AXEL I more and more think we should deprecate the branch concept. With the changes we have done, it is more awkward now.
.. code-block:: python

    import names as n

    from qdesignoptimizer.utils.names_parameters import (
        FREQ, KAPPA, param, param_nonlin,
    )

    PARAM_TARGETS = {
    param(n.QUBIT_1, FREQ): 4e9,
    param(n.RESONATOR_1, FREQ): 6e9,
    param(n.RESONATOR_1, KAPPA): 1e6,
    param_nonlin(n.QUBIT_1, n.QUBIT_1): 200e6,  # Qubit anharmonicity
    param_nonlin(n.QUBIT_1, n.RESONATOR_1): 1e6,  # Qubit resonaotr chi

}


Mini Studies
------------

The core idea of a ``MiniStudy`` is to break down your quantum chip into smaller problems which are more tractable to simulate on a classical computer, (un?)fortunately brute forcing quantum mechanics seems to be hard. However, if you chip is not too large, you might be able to optimize your full chip using a single ``MiniStudy``. The full class documentation is to be found in src/qdesignoptimizer/design_analysis_types.py. Below is a minimal example for a mini study setup of a qubit-resonator system coupled to a transmission line.

.. code-block:: python

    import name as n
    from qdesignoptimizer.design_analysis_types import MiniStudy
    from qdesignoptimizer.utils.utils_design_variables import junction_setup

    MiniStudy(
        qiskit_component_names=[
            n.name_mode(qubit),
            n.name_mode(resonator),
            n.name_tee(group),
        ],
        port_list=[
            (n.name_tee(group), "prime_end", 50),
            (n.name_tee(group), "prime_start", 50),
        ],
        open_pins=[],
        modes=[qubit, resonator],
        jj_setup={**junction_setup(qubit)},
        design_name="get_mini_study_qb_res",
        adjustment_rate=1,
        build_fine_mesh=False,
        **CONVERGENCE
        )

.. caution:: Important is the ordering of the mode frequencies from lowest to highest, and need to match the order of the modes in the HFSS eigenmode simulation.


Plot Settings
-------------
To visualize the progress of the optimizer the evolution of the parameter targets can easily be plotted by a few settings.
A minimal example looks like this:
# TODO AXEL Lukas you mentioned that this failed for you if you don't have the same number of plots in each panel, that should not be the case, is it?

.. code-block:: python

  from qdesignoptimizer.utils.sim_plot_progress import OptPltSet

  PLOT_SETTINGS = {
      "RES": [
          OptPltSet(dc.ITERATION, dc.RES_FREQ),
          OptPltSet(dc.ITERATION, dc.RES_KAPPA),
      ],
      "QUBIT": [
          OptPltSet(dc.ITERATION, dc.QUBIT_FREQ),
          OptPltSet(dc.ITERATION, dc.RES_QUBIT_CHI),
      ],
  }


Optimization Workflow
---------------------
Finally, the user can run the optimization. We suggest to initially optimize every component for their parameter targets (TODO AXEL I usually dont do it like this but rather put some extremer values to be sure that the mode order is correct such that I dont have to create so many ministudies, and for branch-sets of branches I think is not usually what you would do, usually you just pick components at the edges of two branches instead of simulating whole branches), then to optimize for the branch and then for sets of branches.
A minimal example can look like this:

.. code-block:: python

  MINI_STUDY_BRANCH = 0
  MINI_STUDY = ms.get_mini_study_qb_res(branch=MINI_STUDY_BRANCH)
  RENDER_QISKIT_METAL = lambda design: render_qiskit_metal_design(design, gui)

  ################# optimization targets ##############
  opt_targets = [get_opt_target_res_freq_via_length(branch)]

  design_analysis_state = DesignAnalysisState(design, RENDER_QISKIT_METAL, pt.PARAM_TARGETS)
  design_analysis = DesignAnalysis(
      design_analysis_state,
      mini_study=MINI_STUDY,
      opt_targets=opt_targets,
      print_progress=True,
      save_path=CHIP_NAME + "_" + time.strftime("%Y%m%d-%H%M%S"),
      update_parameters = True,
      plot_settings=ps.PLOT_SETTINGS,
      )

  nbr_runs = 10
  nbr_passes = 15  # High number of passes is needed for accurate resuls, but keeping it low can be usedful when developing your design
  delta_f = 0.001
  for i in range(nbr_runs):
      design_analysis.update_nbr_passes(nbr_passes)
      design_analysis.update_delta_f(delta_f)
      design_analysis.optimize_target({}, {})


The optimizer outputs a ``.npy`` file with the target parameters and design variables evaluated after every iteration. In addition, the optimizer can output a new ``.json`` file with the updated design parameters and a snapshot of the qiskit-metal gui to visually follow the progress. The user can also choose to update the initial ``design_variables.json`` file by running ``design_analysis.overwrite_parameters()``.

.. caution:: The design analysis can get stuck on the EPR diagonalization step. We noticed that the problem can be mitigated by choosing a larger number of passes, e.g. 6.
