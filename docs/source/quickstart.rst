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

On this page, we will explain how to set up each of these files.

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


.. caution:: Independent optimization targets are those which include. In contrast, dependent optimization targets correspond to ...


Mini Studies
------------

Target Parameters
-----------------


Plot Settings
-------------


Optimization Workflow
---------------------

.. caution:: The design analysis can get stuck on the diagonalization. We noticed that the problem can be mitigated by choosing a larger number of passes, e.g. 6. 