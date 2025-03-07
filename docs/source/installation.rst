.. _installation:

============
Installation
============
This installation guide will help you install the package QDesignOptimizer on your system.

Installation with pypip
========================
The user can install the package QDesignOptimizer via the python package manager pip. 

.. code-block:: bash

    pip install qdesignoptimizer

Installation from github repository
===================================
The user can install the QDesignOptimizer directly from the github repository. To do so, follow the steps below:


Clone the Repository
--------------------
First, clone the repository and navigate to its directory:

.. code-block:: bash

    git clone https://github.com/202Q-lab/QDesignOptimizer
    cd QDesignOptimizer

Create a Virtual Environment
----------------------------
It is strongly recommended that you install the project in a separate virtual environment, which must use python3.10. This project is packaged with `poetry <https://python-poetry.org/>`, which handles the environment setup for you. If you instead use conda, you can create the environment from the provided environment.yml file:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate qdesignenv

You can also create the environment using venv:

.. code-block:: bash

    # Create new virtual environment
    python3.10 -m venv qdesignenv

    # Activate the environment
    # On Linux/MacOS:
    source qdesignenv/bin/activate
    # On Windows:
    qdesign-env\\Scripts\\activate

    # Verify Python version
    python --version
    # Install poetry if not already available
    pip install poetry


User Installation
-----------------
For regular users, install the project with its dependencies and Qiskit Metal:

.. code-block:: bash

    poetry install
    pip install --no-deps qiskit-metal==0.1.5

Developer Installation
----------------------
For developers, who want to contribute to the project, install with additional development dependencies:

.. code-block:: bash

    poetry install --with docs,analysis
    pip install --no-deps qiskit-metal==0.1.5
    pre-commit install

This will install:

- All project dependencies
- Documentation tools
- Analysis and testing tools
- Pre-commit hooks for code quality

In order to build the documentation yourself, you also need to install `pandoc <https://pandoc.org/>`_ (And you probably need to restart your PC to set the path variables correctly). You can build the documentation by running ``poetry run sphinx-build -b html docs/source docs/_build/htm``.


Installation of Ansys HFSS
==========================
This version of qdesignoptimizer has been tested with Ansys Electronics Desktop 2021 R2.
