.. _installation:

Installation
============

This guide will help you install QDesignOptimizer on your system.

Clone the Repository
--------------------

First, clone the repository and navigate to its directory:

.. code-block:: bash

    git clone https://github.com/202Q-lab/QDesignOptimizer
    cd QDesignOptimizer

Create a Virtual Environment
----------------------------
It is strongly recommended that you install the project in a separate virtual environment, which must use python3.10. If you use conda, you can create it from the provided environment.yml file

.. code-block:: bash

    conda env create -f environment.yml
    conda activate qdesign_env

You can also create it using venv:

.. code-block:: bash

    # Create new virtual environment
    python3.10 -m venv qdesign_env

    # Activate the environment
    # On Linux/MacOS:
    source qdesign-env/bin/activate
    # On Windows:
    qdesign-env\\Scripts\\activate

    # Verify Python version
    python --version
    # For developers: install poetry
    pip install poetry


User Installation
-----------------

For regular users, install the project with its dependencies and Qiskit Metal:

.. code-block:: bash

    poetry install
    pip install --no-deps qiskit-metal==0.1.5

Developer Installation
----------------------

For developers who want to contribute to the project, install with additional development dependencies:

.. code-block:: bash

    poetry install --with docs,analysis
    pip install --no-deps qiskit-metal==0.1.5
    # ignore pre-commit install

This will install:

- All project dependencies
- Documentation tools
- Analysis and testing tools
- Pre-commit hooks for code quality

In order to build the documentation yourself, you also need to install `pandoc <https://pandoc.org/>`_.


Install Ansys
-------------

This version of qdesignoptimizer has been tested with Ansys Electronics Desktop 2021 R2.
