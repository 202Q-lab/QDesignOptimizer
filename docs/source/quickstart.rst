==============
QDO quickstart
==============
This quickstart guide takes the example of a single transmon-resonator system to examplify the setup of the optimization project. Based on this concept more complex optimization problems can be constructed.

Installation
============
For the installation we refer back to the installation guide :ref:`installation`.

Project structure
=================
Every optimization project requires a set of files defined the optimization problem:

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

Project setup
=============

Mode Names and Component Names
------------------------------
| For mode names we suggest a naming convention of the form ``mode_name_identifier`` composed by the convenience function ``mode(mode_type, identifier)`` in ``utils.names_parameters.py``, for example ``qubit_1``. The mode type can for example be ``resonator``, ``qubit``, ``cavity``, or ``coupler``. As identifier we suggest a count of the component group or of the component in a group of components.
| For qiskit-metal component names we suggest a naming convention of the form ``name_identifier``, for example ``name_qubit_1`` or ``name_tee_1``. The identifier can refer to mode name such as ``qubit_1`` or ``resonator_1``. A collection of common qiskit-metal component names can be directly called from ``utils.names_qiskit_components`` or custom-made in the project file ``names.py``.


Design Variable Names and Values
--------------------------------
| Design variable names are string identifiers (for geometric lengths, Josephson junction inductances etc.) specified in the qiskit-metal design. The design variables can be varied by the optimizer during the optimization to reach the target parameters. For the design variable names we suggest a naming convention of the form ``design_var_`` followed by an identifier which indicates what the design variable controls, for example ``design_var_length_resonator_1``. A collection of common design variable names can be directly called from ``utils.names_design_variables.py``. User specific component names can be added in and called from ``names.py`` in the project folder.
| To render the qiskit-metal design, the user must provide initial values based on a sensible guess for all design variables, which by convention are written in ``design_variables.json`` in the project folder and provided to the optimizer.

Design
------
| The user-specified qiskit-metal design can be created as usual following the qiskit-metal guide line using general or custom-made circuit components. Any dimension of a component which should be varied during the optimization must be provided as a design variable identifier.
| As a minimal example we can look at the definition of a ``RouteMeander`` resonator. In this example, we show that the design variable substitues the otherwise commonly static design definition for the total resonator length.

.. code-block:: python

    import design_variable_names as u
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

    resonator_options = dict(
        total_length=n.design_var_length(resonator),
        trace_width='20um',
        trace_gap='20um',
    )

    RouteMeander(design, n.name_mode(resonator), options=resonator_options)

Finally, the design can be instantiated by the ``create_chip_base`` method and rendered with the components and the design variables. A wrapper function (by convention called ``render_qiskit_metal_design``), must be created such that it can be passed into the optimizer. A minimal example looks like this:

.. code-block:: python

    import design as d
    import names as n
    from qdesignoptimizer.utils.utils_design import create_chip_base, ChipType

    CHIP_NAME = "transmon_chip"
    OPEN_GUI = True
    chip_type = ChipType(size_x="10mm", size_y="10mm", size_z="-300um")
    design, gui = create_chip_base(chip_name=CHIP_NAME, chip_type=chip_type, open_gui=OPEN_GUI)
    n.add_design_variables_to_design(design, dv)

    def render_qiskit_metal_design(design, gui):
        d.add_transmon_plus_resonator(design, group=n.NBR_1)

        gui.rebuild()
        gui.autoscale()

    render_qiskit_metal_design(design, gui)
    # This line will render the qiskit design in the gui, which is useful when developing the design.

.. _opttarget:

Optimization Target
--------------------
| The optimization target ``OptTarget`` is a required core component of the qdesignoptimizer. It relates the parameter target (e.g. frequency, kappa, capacitance, or Purcell limited T1) with the involved modes (e.g. ``resonator`` or ``qubit``), the design variable (e.g. ``design_var_length_resonator_1``) and the physical relation used during optimization (e.g. ``1/design_var_length_resonator_1`` in case of the resonator frequency).
| One ``OptTarget`` must be created for each target parameter the user wants to optimize for. The names of the involved eigenmodes and parameter names is by convention called from ``names.py`` in the project folder.
| The full class documentation can be found in src/qdesignoptimizer/design_analysis_types.py.
| A minimal example for the resonator length can look like this:

.. code-block:: python

    from qdesignoptimizer.design_analysis_types import OptTarget
    import design_constants as dc
    import design_variable_names as u
    def get_opt_target_res_freq_via_length(
        resonator: Mode,
        design_var_res_length: Callable = n.design_var_length,
        ) -> OptTarget:

    return OptTarget(
        target_param_type=n.FREQ,
        involved_modes=[resonator],
        design_var=design_var_res_length(resonator),
        design_var_constraint={"larger_than": "500um", "smaller_than": "15000um"},
        prop_to=lambda p, v: 1 / v[design_var_res_length(resonator)],
        independent_target=True,
    )

More involved and dependent physical relations can be formulated using parameters ``p`` and design variables ``v`` in the propotionality statement of the ``OptTarget``. An example for a more detailed relation can be formulated for the nonlinear parameter :math:`\chi`:

.. code:: python

    prop_to = lambda p, v: np.abs(v[design_var_res_qb_coupl_length(resonator, qubit)] / v[design_var_qubit_width(qubit)] * p[param_nonlin(qubit, qubit)] / (p[param(qubit, FREQ)] - p[param(resonator, FREQ)] - p[param_nonlin(qubit, qubit)] ))

.. caution:: Ensure that the units of the design variable match the unit of the contraint in the optimization target and the parameters in the propotionality statement prop_to. For consistency we suggest to use the units :math:`um` for measures of length, :math:`nH` for inductances and :math:`fF` for capacitances.

.. _relationtable:

Physical relation
-----------------

One strength of the qdesignoptimizer arises from the integration of physical relations between the design variable and the parameter targets, which boosts the efficiency of the optimization. Note that the ``OptTarget`` only requires an expression which is proportional to the target quantity, since it only uses relative values in the update step. Hence, the user only need to provide the part of the function which varies and to the level of detail which is known to the user. The more accurate the user specified model is, the faster and more robust the optimizer will be. The table below contains an example set of suggested physical relations for the optimization targets for Hamiltonian and dissipative parameters in a dispersively coupled qubit-resonator cQED system.:

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

.. caution::  An OptTarget can be marked as independent_target=True if the target only depends on a single design variable and not on any system parameter. This allows the optimizer to solve this OptTarget independently, making it faster and more robust. If a criteria of independence is not fulfilled, the OptTarget must be independent_target=False (as the default).

Parameter Targets
-----------------
| The parameter targets are specified in a ``dict`` per target parameter. Three types of of parameter targets can be defined, (1) parameters ``param`` with mode and parameter type, (2) nonlinear parameters ``param_nonlin`` between two modes, and (3) capacitance targets ``param_capacitance`` between two component names. Note that the nonlinear parameters are self-Kerr or anharmonicity :math:`\alpha` and cross-Kerr or :math:`\chi` parameters. They follow the bosonic definition of qiskit-metal.
| A minimal example for a single qubit-resonator system may look like this:

.. code-block:: python

    import names as n

    from qdesignoptimizer.utils.names_parameters import (
    param,
    param_capacitance,
    param_nonlin,
    )

    PARAM_TARGETS = {
        param(n.QUBIT_1, n.FREQ): 4e9,
        param(n.QUBIT_1, n.PURCELL_LIMIT_T1): 20e-3,
        param(n.RESONATOR_1, n.FREQ): 6e9,
        param(n.RESONATOR_1, n.KAPPA): 1e6,
        param_nonlin(n.QUBIT_1, n.QUBIT_1): 200e6,  # Qubit anharmonicity
        param_nonlin(n.QUBIT_1, n.RESONATOR_1): 1e6,  # Qubit resonaotr chi
        param_capacitance("prime_cpw_name_tee1", "second_cpw_name_tee1"): -3, # fF
    }

.. caution:: Make sure that all frequencies and rates are defined in units of Hz.

Mini Studies
------------
| The idea of a ``MiniStudy`` is to break down the quantum chip design into smaller problems which are more tractable to simulate on a classical computer, (un?)fortunately brute forcing quantum mechanics seems to be hard. However, if your chip design is not too large, you might be able to optimize your full chip design using a single ``MiniStudy``.
| The full class documentation can be found in src/qdesignoptimizer/design_analysis_types.py.
| Below is a minimal example for a mini study setup of a qubit-resonator system coupled to a transmission line.

.. code-block:: python

    import name as n
    from qdesignoptimizer.design_analysis_types import MiniStudy
    from qdesignoptimizer.utils.utils_design_variables import junction_setup

    CONVERGENCE = dict(nbr_passes=7, delta_f=0.03)

    MiniStudy(
        qiskit_component_names=[
            n.name_mode(n.QUBIT_1),
            n.name_mode(n.RESONATOR_1),
            n.name_tee(n.NBR_1),
        ],
        port_list=[
            (n.name_tee(n.NBR_1), "prime_end", 50),
            (n.name_tee(n.NBR_1), "prime_start", 50),
        ],
        open_pins=[],
        modes=[n.QUBIT_1, n.RESONATOR_1],
        jj_setup={**junction_setup(n.RESONATOR_1)},
        design_name="get_mini_study_qb_res",
        adjustment_rate=1,
        build_fine_mesh=False,
        **CONVERGENCE
        )

.. caution:: The order of modes defined in the MiniStudy must match the order of modes resulting from the HFSS eigenmode simulation, which goes from lowest to highest frequency.

Plot Settings
-------------
| To visualize the progress of the optimization, the evolution of the parameter targets can be plotted in a custom way.
| A minimal example looks like this:


.. code-block:: python

    from qdesignoptimizer.utils.sim_plot_progress import OptPltSet
    from qdesignoptimizer.utils.names_parameters import (
        param,
        param_capacitance,
        param_nonlin,
        )

    PLOT_SETTINGS = {
        "RES": [
            OptPltSet(n.ITERATION, param(n.RESONATOR_1, n.FREQ), y_label="RES Freq (Hz)"),
            OptPltSet(n.design_var_length(n.RESONATOR_1), param(n.RESONATOR_1, n.FREQ), y_label="RES Freq (Hz)"),
            OptPltSet(n.ITERATION, param(n.RESONATOR_1, n.KAPPA), y_label="RES Kappa (Hz)"),
            OptPltSet(n.ITERATION, param_nonlin(n.RESONATOR_1, n.RESONATOR_1), y_label="RES Kerr (Hz)"),
        ],
        "QUBIT": [
            OptPltSet(n.ITERATION, param(n.QUBIT_1, n.FREQ), y_label="QB Freq (Hz)"),
            OptPltSet(n.ITERATION, param_nonlin(n.QUBIT_1, n.QUBIT_1), y_label="QB Anharm. (Hz)"
            ),
        ],
        "COUPLINGS": [
            OptPltSet(n.ITERATION, param_nonlin(n.RESONATOR_1, n.QUBIT_1), y_label="RES-QB Chi (Hz)"),
        ],
    }

Optimization Workflow
---------------------
| Once the optimization problem has been set up, the user can start the optimization. We suggest to break the entire optimization problem down into smaller optimization problems defined as mini studies of groups of qiskit-metal components (e.g. a set of ``resonator``, ``qubit``, and ``feedline`` as a tile of a larger chip design). Subsequently, the user can optimize linking qiskit-metal components between the groups that have been studied initially (e.g. ``qubit_1``, ``coupler``, ``qubit_2``).
| A minimal example can look like this:

.. code-block:: python

    # select MiniStudy
    MINI_STUDY_GROUP = n.NBR_1
    MINI_STUDY = ms.get_mini_study_qb_res(group=MINI_STUDY_GROUP)
    RENDER_QISKIT_METAL = lambda design: render_qiskit_metal_design(design, gui)

    # select OptTarget
    opt_targets = [get_opt_target_res_freq_via_length(branch)]

    # initialization
    design_analysis_state = DesignAnalysisState(
        design, RENDER_QISKIT_METAL, pt.PARAM_TARGETS
    )
    design_analysis = DesignAnalysis(
        design_analysis_state,
        mini_study=MINI_STUDY,
        opt_targets=opt_targets,
        save_path="out/" + CHIP_NAME + "_" + time.strftime("%Y%m%d-%H%M%S"),
        update_design_variables=False,
        plot_settings=ps.PLOT_SETTINGS,
    )

    # optimization
    group_runs = 10
    group_passes = 14
    delta_f = 0.001
    for i in range(group_runs):
        design_analysis.update_nbr_passes(group_passes)
        design_analysis.update_delta_f(delta_f)
        design_analysis.optimize_target({}, {})
        design_analysis.screenshot(gui=gui, run=i)


The optimizer outputs a ``.npy`` file with the target parameters and design variables evaluated after each iteration. In addition, the optimizer can output a new ``.json`` file with the updated design parameters and a snapshot of the qiskit-metal gui to visually follow the progress. The user can also choose to update the initial ``design_variables.json`` file by running ``design_analysis.overwrite_parameters()``.
