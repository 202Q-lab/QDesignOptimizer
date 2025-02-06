Optimizer 101
===========

This section contains information about the workflow of the optimizer.

Motivation
----------
Tuning up a superconducting quantum chip often involves lenghty simulations where the user manually updates the design variables, such as capacitance lengths and Josephson junction sizes, in order to reach the target quantities of the circuit, such as frequencies, decay rates and coupling strengths. The QDesignOptimizer package strongly reduces the need for manual intervention by automating the simulation and optimization of the design variables. The optimization cycle first runs a detailed electro-magnetic HFSS simulation combined with EPR analysis to estimate the current values of the circuit quantities. The second step solves a nonlinear approximate model based on the user's physical knowledge about the circuit, which estimates how the design variables should be changed to reach the target. The flexibility in the QDesignOptimizer setup allows for efficient investigation of chip subsystems, since the simulation and approximate physical model are dynamically compiled from the user settings. 

.. toctree::
   :maxdepth: 2

   concept


References
----------
More information about the functionality of the optimizer and the setup of this design optimizer package can be found in the publication tbp. 
   