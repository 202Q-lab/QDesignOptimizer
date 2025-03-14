.. _qdesignoptimizer:

=============
Optimizer 101
=============

This section contains information about the workflow of the QDesignOptimizer (QDO).

Why QDO?
=============
Tuning up a superconducting quantum chip often involves lenghty simulations where the user manually updates the design variables, such as capacitance lengths and Josephson junction sizes, to reach the parameter targets of the circuit, such as frequencies, decay rates and coupling strengths. The QDesignOptimizer package strongly reduces the need for manual intervention by automating the simulation and optimization of the design variables. The optimization cycle first runs a detailed and often time-taking electro-magnetic `Ansys HFSS <https://www.ansys.com/products/electronics/ansys-hfss>`_ simulation [#f1]_ combined with `Energy Participation Ratio (EPR) <https://pyepr-docs.readthedocs.io/en/latest/>`_ analysis [#f2]_ to estimate the current values of the circuit parameters, integrated together with design framework `Qiskit Metal <https://qiskit-community.github.io/qiskit-metal/>`_ [#f3]_. The second step solves a nonlinear approximate model based on the user's physical knowledge about the circuit, which estimates how the design variables should be changed to reach the target. The flexibility in the QDesignOptimizer setup allows for the efficient investigation of chip subsystems since the simulation and approximate physical model are dynamically compiled from the user settings.

QDO Concept
=============
The optimiation framework allows for a systematic and automated optimization of quantum chip designs based on physical models including:

1. Eigenmode simulations 
2. Capacitance simulations
3. Energy participation ratio analysis 

The workflow of the QDesignOptimizer follows a three-step logic, which is also shown in the figure below:

1. Problem formulation
2. Optimization based on physical models
3. Validation and results

In short, in the problem formulation the user specifies the parameter targets, an initial guess for the design variables, and the optimization targets holding the approximate physical relationships between the parameters and variables. The optimization step is an iterative process that begins with a detailed electromagnetic simulation and energy participation ratio analysis for the current design variables, which provide the current values of the parameters. The approximate nonlinear model specified in the optimization targets is then solved, suggesting a good approximation to the optimal design variables, leading to convergence after a few iterations.
| Further information on the setup can be found in :ref:`userguide`. 

.. figure:: optimizationflow.png
   :width: 450px
   :scale: 100%
   :alt: Example image
   :align: center

Problem Formulation
-------------------

The general idea is to identify :math:`N` design variables :math:`\overrightarrow{V}=\{V_1, ..., V_N\}`, which can be varied so that the targets of the :math:`N` parameters :math:`\overrightarrow{Q}=\{Q_1, ..., Q_N\}` can be reached. These definitions are done by setting up one :ref:`opttarget` for each parameter, which specifies the physical, generally nonlinear relation of how the parameter depends on the varying design variables as well as (other) parameters, i.e.,

.. math::

   Q_i\propto F_i\left(\underset{\textrm{excl.} Q_i}{ \overrightarrow{Q}},  \overrightarrow{V}\right)

As an initial guess for the optimization, the user will provide sensible values for the design variables :math:`\overrightarrow{V}`.


Simulate Design Variables
-------------------------

To obtain knowledge of which values of the parameters the current design variables result in, the QDesignOptimizer runs a detailed electromagnetic simulation (by choice an eigenmode and/or capacitance simulation) in `Ansys HFSS <https://www.ansys.com/products/electronics/ansys-hfss>`_ and performs an energy-participation ratio (`pyEPR <https://pyepr-docs.readthedocs.io/en/latest/>`_) analysis. The parameters :math:`\overrightarrow{Q}^{k}`, in the k-th iteration, are then extracted from the simulation and analysis results.

Update Design Variables
-------------------------

In update step number :math:`k`, the optimizer updates the design variables from the ones used in the simulation :math:`\overrightarrow{V}^{k}` to :math:`\overrightarrow{V}^{k+1}`, such that :math:`\overrightarrow{Q}_i^{k+1}` gets close to the target value :math:`Q_i^{target}`. Due to the assumption of proportionality between :math:`Q_i` and :math:`F_i`, the updated parameter is estimated as:

.. math::

   \tilde Q_i^{k+1} = Q_i^{k} \frac{F_i(\overrightarrow{\tilde Q}^{k+1},\overrightarrow{V}^{k+1})}{F_i(\overrightarrow{Q}^k,\overrightarrow{V}^k)}.

Note that we here distinguish between the parameter :math:`\overrightarrow{\tilde Q}` estimated by the approximate model and the simulated parameter :math:`\overrightarrow{Q}`, which is typically more accurate. The more accurate the physical model is, which we set up in the equation above, the smaller is the discrepancy :math:`|| \overrightarrow{Q}- \overrightarrow{\tilde Q}||`, and the faster the optimization converges.

To obtain the updated design variables :math:`\overrightarrow{V}^{k+1}`, the QDesignOptimizer minimizes the cost function:

.. math::

   C = \sum_{i=1}^N\left|\frac{\tilde Q_i^{k+1}}{Q_i^{target}} - 1\right|^2

by finding the optimal :math:`\overrightarrow{V}^{k+1}`. If the problem is correctly formulated, the minimization will reach :math:`\tilde Q_i^{k+1} = Q_i^{target}` for all :math:`k=1,...,N` targets in the optimization. However, the QDesignOptimizer assumes that parameters, which are not associated with an :ref:`opttarget`, will not be affected by the changed design variables, i.e., :math:`\tilde Q_i^{k+1} = Q_i^{k}` for :math:`k>N`, if the system contains more parameters than targets.

These relations for :math:`\tilde Q_i^{k+1}` simplify parameter updates to only depend on:

- The values of the parameters in the previous step,
- The target values, and
- The design variables.

One of the main assumptions, which the QDesignOptimizer takes advantage of is that, if the approximate model incorporates the correct general trends of the physical relationships, the optimization will converge to the target. Hence, there is no need for the user to specify a very precise physical model, but the more the user knows about the physics, the faster and more robust the optimizer will be.

Separating physical dependencies by design
-------------------------------------------

| We recommend to create a design which separates the physical dependence between parameter targets and their intendet design variables. As a result, the user can specify more easily a nonlinear model that approximates the physical dependences of the design well. This nonlinear model is the input to the :ref:`opttarget`. Note that, the user does not need to decouple the physics of the system as long as the user can model the coupled system by nonlinear equations well. In many cases, it might not even be necessary to develop a very precise mode. An inaccurate model capturing the gradient is often sufficient, if the optimizer takes small update steps, which can be set by the update rate of the optimizer. However, the user might compromise on convergence.  
| For example, if we define the :math:`l_{res-tl}` coupling length such that it does not affect the total length of the resonator, we approximately decouple the optimization of :math:`f_{res}` and :math:`\kappa_{res}`. In this example, given that we decoupled the physical relation between coupling strenght and frequency, a simple decoupled nonlinear model is a good approximation of the system.


Independent Variables
-----------------------

The number of independent design variables :math:`N` must match the number of parameters that have a target in the optimization. In the example discussed in :ref:`qickstart`, we consider the :math:`N=5` parameters specified in table under :ref:`relationtable`, where the corresponding five design variables are:

- Resonator length :math:`l_{res}`
- Qubit Josephson junction inductance :math:`L_{qb}`
- Qubit width :math:`w_{qb}`
- Resonator-qubit coupling width :math:`w_{res-qb}`
- Resonator to transmission line coupling length :math:`l_{res-tl}`


Factorization of Update Step
----------------------------

| The nonlinear minimization step is simplified by exploiting the independence of some design variables in the physical relations. Involving this factorization, we can decompose the original N-dimensional optimization problem into a sequence of lower-dimensional subproblems, which significantly reduces the computational complexity and can be solved faster.
| Specifically in the example discussed in :ref:`qickstart` we observe that: 

- The resonance frequency of the resonator :math:`f_{res}` depends solely on the resonator length :math:`l_{res}`
- The coupling of the resonator to the feedline :math:`\kappa_{res}` depends solely on the resonator to feedline coupling length :math:`l_{res-tl}`
- The qubit capacitance energy :math:`f_{qb}` is influenced only by the qubit width :math:`w_{qb}`.

Instead of minimizing all parameters simultaneously, the optimizer first solve the following one-dimensional optimization problems to obtain the updated design variables:

- Determine :math:`l_{res}^{k+1}` by minimizing the cost function with respect to :math:`(f_{res}, l_{res})`.
- Determine :math:`l_{res-tl}^{k+1}` by minimizing the cost function with respect to :math:`(\kappa_{res}, l_{res-tl})`.
- Determine :math:`w_{qb}^{k+1}` by minimizing the cost function with respect to :math:`(f_{qb}, w_{qb})`.

Once these one-dimensional optimizations are complete, we solve the remaining two-dimensional problem involving:

- Determine :math:`\chi_{qb-res}^{k+1}` by solving for :math:`(f_{qb}, \chi, L_{qb}, w_{res-qb})`

| Instead of solving a full five-dimensional problem at once, we handle subproblems of dimensions 1, 1, 1, and 2, which are computationally more efficient. 



.. rubric:: Footnotes

.. [#f1] `Ansys HFSS <https://www.ansys.com/products/electronics/ansys-hfss>`_ is a proprietary, multipurpose, full wave 3D electromagnetic (EM) simulation software for designing and simulating high-frequency electronic products such as antennas, components, interconnects, connectors, ICs, and PCBs.
.. [#f2] `pyEPR <https://pyepr-docs.readthedocs.io/en/latest/>`_ is an open-source library providing automated analysis and design of quantum microwave devices. This package is based on the publication Minev, Z.K., Leghtas, Z., Mundhada, S.O. et al. Energy-participation quantization of Josephson circuits. npj Quantum Inf 7, 131 (2021).
.. [#f3] `Qiskit Metal <https://qiskit-community.github.io/qiskit-metal/>`_ is an open-source framework (and library) for the design of superconducting quantum chips and devices.

