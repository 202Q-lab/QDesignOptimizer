Contributing
========================
QDesignOptimizer is an open source project. If you come across a missing feature that could benefit a wider audience, we encourage you to open an issue and reach out to the developers. Your insights and contributions help shape and improve the project for everyone!

Testing the tutorial optimizations
----------------------------------
The tests in ``tests/integration`` let you change the library without running the
tutorial optimizations by hand. They need no Ansys installation:

* Every tutorial notebook in ``tutorials/examples_*`` is executed end-to-end. The
  qiskit-metal design is built for real, while HFSS, pyEPR and Q3D are replaced by
  recording fakes that answer with a deterministic model of the chip.
* Each run is compared against a snapshot in ``tests/integration/snapshots``. The
  snapshot contains every simulator call (passes, delta_f, variables, junctions,
  truncations, Q3D setup, surface rendering, ...) and, for every iteration, the
  design variables, parameters, eigenmode results, cross-Kerr matrices,
  capacitance matrices and optimizer groups.
* Every example must still converge, and every parameter reported by
  ``DesignAnalysis`` must equal what the fake simulator produced.
* ``test_design_analysis_scenarios.py`` covers features that no notebook uses
  (non-default settings, linear junctions, adjustment rate, constraints, ...), and
  ``test_scalability_example.py`` runs a scaled-down version of the cluster example.

If your change intentionally alters what is sent to the simulators or how the
optimizer updates the design, regenerate the snapshots and review the diff before
committing:

.. code-block:: bash

    QDO_UPDATE_SNAPSHOTS=1 pytest tests/integration

New tutorial notebooks must be registered in ``EXAMPLE_CASES`` in
``tests/integration/conftest.py``.

Publishing a new release to PyPI
---------------------------------
If you are taking over maintenance of the repository and need to publish a new
release, follow the step-by-step checklist in
`.github/RELEASING.md <https://github.com/202Q-lab/QDesignOptimizer/blob/main/.github/RELEASING.md>`_.
It covers merging PRs, bumping the version, running tests, using the TestPyPI
dry-run workflow, and creating the GitHub Release that triggers the live publish.

The latest stable version is published as a pip-installable release
(``pip install qdesignoptimizer``); see the :ref:`installation` guide for details.
