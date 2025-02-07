QDO quickstart
==============

Project structure
-----------------

Every optimizer project requires a set of files:

.. code-block::

    project_root/
    ├── design_variables.py
    ├── design.py
    ├── main.ipynb
    ├── mini_studies.py
    ├── optimization_targets.py
    ├── plot_settings.py
    ├── target_parameters.py

On this page, we will explain how to set up each of these files. To illustrate the concept of the optimizer setup, we will follow the example of a single coupled qubit-resonator system. This minimal example can be found in projects/example_single_transmon. 

Branches
--------
The concept of branches is important for the efficient optimization of larger superconducting quantum circuits. In our definition a branch can be any set of circuit elements with corresponding eigenmode. 


Design Variables
-----------------




Design
------

.. caution:: In the design, the correct use of the design variables is important. Use ... 


Optimization Targets
--------------------

The table below contains a set of suggested, but not limited list of optimization targets for Hamiltonian and dissipative quantities:

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



.. caution::  Mark independent_target=True if the target only depends on a single design variable and not on any system parameter. This allows the optimizer to solve this OptTarget independently, making it faster and more robust.

.. caution:: Ensure that the units of the design variable matches the unit of the contrain in the optimization target and the parameters in the propotionality statement prop_to. For better consistency use the units :math:`um` for measures of length, :math:`nH` for inductances and :math:`fF` for capacitances.


Mini Studies
------------

The ``MiniStudy`` class object defines a study configuration with various parameters. The full class documentation in be found in src/qdesignoptimizer/design_analysis_types.py. Below is a conceptual example for a mini study setup of resonator coupled to a transmission line:

.. code-block:: python

    MiniStudy(
        component_names=["COMPONENT_NAME"],
        port_list=[
            ("COMPONENT_NAME", "PORT START", 50),
            ("COMPONENT_NAME", "PORT END", 50),
        ],
        open_pins=[],
        mode_freqs=[
            ("BRANCH NUMBER", "EIGEN FREQUENCY"),
        ],
        jj_var="DESIGN VARIABLE JUNCTION INDUCTANCE",
        jj_setup={**junction_setup(dv.name_qb(branch))},
        design_name="get_mini_study_res",
        adjustment_rate=0.8,
        **CONVERGENCE
    )

This study involves:

- **Component Names**: Defines the qubit, resonator, and tee junction components.
- **Port List**: Specifies the connection points and their impedances.
- **Open Pins**: Lists any unconnected pins (empty in this case).
- **Mode Frequencies**: Defines frequency modes for qubit and resonator.
- **Junction Variables**: Sets Josephson junction parameters.
- **Junction Setup**: Configures the junction for the qubit.
- **Design Name**: Identifies the study.
- **Adjustment Rate**: Controls convergence speed.

.. tip:: Wrap the mini study object in a function 

This setup ensures an optimized mini study for circuit analysis and design.

Target Parameters
-----------------


Plot Settings
-------------


Optimization Workflow
---------------------

.. caution:: The design analysis can get stuck on the diagonalization. We noticed that the problem can be mitigated by choosing a larger number of passes, e.g. 6. 