.. _qdesignoptimizer:

The Optimizer
=============

This section contains information about the workflow of the QDesignOptimizer (QDO).

What is QDO?
------------
Tuning up a superconducting quantum chip often involves lenghty simulations where the user manually updates the design variables, such as capacitance lengths and Josephson junction sizes, in order to reach the parameter targets of the circuit, such as frequencies, decay rates and coupling strengths. The QDesignOptimizer package strongly reduces the need for manual intervention by automating the simulation and optimization of the design variables. The optimization cycle first runs a detailed electro-magnetic HFSS simulation combined with EPR analysis to estimate the current values of the circuit parameters. The second step solves a nonlinear approximate model based on the user's physical knowledge about the circuit, which estimates how the design variables should be changed to reach the target. The flexibility in the QDesignOptimizer setup allows for efficient investigation of chip subsystems, since the simulation and approximate physical model are dynamically compiled from the user settings.

QDO Concept
-----------
The workflow of the QDesignOptimizer is depicted in the figure below, starting with the user setting up the problem formulation, including:

1. Parameter Targets
2. An initial guess for the design variables
3. Optimization targets holding the approximate physical relationships between the parameters and variables

The optimization step is an iterative process that begins with a detailed electromagnetic simulation and energy participation ratio analysis for the current design variables, which provide the current values of the parameters. The approximate nonlinear model specified in the optimization targets is then solved, suggesting a good approximation to the optimal design variables, leading to convergence after a few iterations.

The workflow of the QDesignOptimizer is depicted in the figure below starting by the user setting up the problem formulation, followed by an iterative optimization of the detailed electro-magnetic simulation.

.. figure:: optimizationflow.png
   :width: 450px
   :scale: 100%
   :alt: Example image
   :align: center

   Caption

Problem Formulation
-------------------

The general idea is to identify :math:`N` design variables :math:`\overrightarrow{V}=\{V_1, ..., V_N\}`, which can be varied so that the targets of the :math:`N` parameters :math:`\overrightarrow{Q}=\{Q_1, ..., Q_N\}` can be reached. These definitions are done by setting up one optimization target (**OptTarget**) for each parameter, which specifies the proportionality relation of how the parameter depends on the varying design variables as well as (other) parameters, i.e.,

.. math::

   Q_i\propto F_i\left(\underset{\textrm{excl.} Q_i}{ \overrightarrow{Q}},  \overrightarrow{V}\right)

As an initial guess for the optimization, the user will provide sensible values for the design variables :math:`\overrightarrow{V}`.


Simulate Design Variables
-------------------------

To obtain knowledge of which values of the parameters the current design variables result in, the QDesignOptimizer runs a detailed electromagnetic simulation in HFSS (potentially also a capacitance matrix) and performs an energy-participation ratio (**EPR**) analysis. The parameters :math:`\overrightarrow{Q}^{k}`, in the k-th iteration, are then extracted from the simulation and analysis results.

Update Design Variables
-------------------------

In update step number :math:`k`, we want to update the design variables from the ones used in the simulation :math:`\overrightarrow{V}^{k}` to :math:`\overrightarrow{V}^{k+1}`, such that :math:`\overrightarrow{Q}_i^{k+1}` gets close to the target value :math:`Q_i^{target}`. Due to the assumption of proportionality between :math:`Q_i` and :math:`F_i`, the updated parameter is estimated as:

.. math::

   \tilde Q_i^{k+1} = Q_i^{k} \frac{F_i(\overrightarrow{\tilde Q}^{k+1},\overrightarrow{V}^{k+1})}{F_i(\overrightarrow{Q}^k,\overrightarrow{V}^k)}.

Here, we distinguish the parameter :math:`\overrightarrow{\tilde Q}` estimated by the approximate modeland the simulated parameter :math:`\overrightarrow{Q}`, which is typically more accurate. The more accurate model we set up in the equation above, the smaller is the discrepancy :math:`|| \overrightarrow{Q}- \overrightarrow{\tilde Q}||`, and the faster the optimization converges.

To obtain the updated design variables :math:`\overrightarrow{V}^{k+1}`, the QDesignOptimizer minimizes the cost function:

.. math::

   C = \sum_{i=1}^N\left|\frac{\tilde Q_i^{k+1}}{Q_i^{target}} - 1\right|^2

by finding the optimal :math:`\overrightarrow{V}^{k+1}`. If the problem is correctly formulated, the minimization will reach :math:`\tilde Q_i^{k+1} = Q_i^{target}` for all :math:`k=1,...,N` targets in the optimization. However, the QDesignOptimizer assumes that parameters not associated with an OptTarget will not be affected by the changed design variables, i.e., :math:`\tilde Q_i^{k+1} = Q_i^{k}` for :math:`k>N`, if the system contains more parameters than targets.

These relations for :math:`\tilde Q_i^{k+1}` simplify parameter update to only depend on:

- The values of the parameters in the previous step,
- The target values, and
- The design variables.

One of the main assumptions which the QDesignOptimizer takes advantage of is that, as long as the approximate model incorporates the correct general trends of the physical relationships, the optimization will converge to the target. Hence, there is no need for the user to specify a very precise physical model, but the more the user knows about the physics, the faster and more robust the optimizer will be.


Independent Variables
-----------------------

The number of independent design variables :math:`N` needs to match the number of parameters that have a target in the optimization. In this example, we consider the :math:`N=5` parameters specified in **Table \ref{parameter_table}**, where the corresponding five design variables are:

- Resonator length :math:`l_{res}`
- Qubit Josephson junction inductance :math:`L_{qb}`
- Qubit width :math:`w_{qb}`
- Resonator-qubit coupling width :math:`w_{res-qb}`
- Resonator to transmission line coupling length :math:`l_{res-tl}`


Factorization of Update Step
----------------------------

The nonlinear minimization step is simplified by noting that the parameters :math:`f_{res}` and :math:`E_c` only depend on :math:`l_{res}` and :math:`w_{qb}`, respectively. Hence, we can reduce the dimension of the minimization problem by running cost function first for the **one-dimensional** problems:

1. :math:`(f_{res}, l_{res})`
2. :math:`(\kappa_{res}, l_{res-tl})`
3. :math:`(f_{qb}, w_{qb})`

to obtain :math:`l_{res}^{k+1}` and :math:`w_{qb}^{k+1}`. Then, we minimize the remaining **two-dimensional** problem for :math:`(f_{qb}, \chi, L_{qb}, w_{res-qb})`.

This way, we solve smaller problems of dimensions **1, 1, 1, and 2** instead of running the full **5-dimensional** problem, which generally takes longer to solve. Whenever possible, it is wise to define design variables that affect only a single parameter independently. For example, if we define the :math:`l_{res-tl}` coupling length such that it does not affect the total length of the resonator, we (approximately) decouple the optimization of :math:`f_{res}` and :math:`\kappa_{res}`.


References
----------
More information about the functionality of the optimizer and the setup of this design optimizer package can be found in the publication tbp.
