.. _installation:

============
Installation
============

This guide will help you install the ``qdesignoptimizer`` package and its dependencies on your system.

Prerequisites
=============

- Python 3.11 or 3.12 (required)
- pip or poetry package manager. We recommend creating a virtual environment by following the steps below.
- Git (for installation from repository)
- Ansys HFSS. ``qdesignoptimizer`` has been tested with Ansys Electronics Desktop 2021 R2 and 2022 R2.

Installation with pip
=====================

The simplest way to install ``qdesignoptimizer`` is via the Python package manager pip:

.. code-block:: bash

    pip install qdesignoptimizer
    pip install --no-deps quantum-metal

Note that Quantum Metal (formerly Qiskit Metal) must be installed separately and without dependencies to make it work properly. The package uses PySide6 (version 6.8+) for the GUI interface. Note that the full installation has been tested with the version quantum-metal==0.5.3.post1. 

Installation from GitHub Repository
===================================

For the latest version or if you want to contribute to development, you can directly clone and install from the GitHub repository.

Cloning the Repository
----------------------

First, clone the repository and navigate to its directory:

.. code-block:: bash

    git clone https://github.com/202Q-lab/QDesignOptimizer
    cd QDesignOptimizer

Creating a Virtual Environment
------------------------------

It is strongly recommended to install the project in a separate virtual environment using our installation guide. You have several options:

**Option 1: Using Poetry (Recommended)**

This project is packaged with `poetry <https://python-poetry.org/>`_, which handles the environment setup for you.

**Option 2: Using Conda**

Create the environment from the provided environment.yml file:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate qdesignenv

Note that the C++ developer tools from the latest distribution of Visual Studio Installer are required for the successful installation of PySide6, which is part of the the environment.yml file. You can download the software from `microsoft <https://visualstudio.microsoft.com/downloads/>`_.

**Option 3: Using venv**

Create and activate a virtual environment using Python's built-in venv module:

.. code-block:: bash

    # Create new virtual environment
    python3.12 -m venv qdesignenv

    # Activate the environment
    # On Linux/MacOS:
    source qdesignenv/bin/activate
    # On Windows:
    qdesignenv\Scripts\activate

    # Verify Python version
    python --version

    # Install poetry if not already available
    pip install poetry

User Installation
-----------------

For regular users, install the project with its dependencies and Quantum Metal:

.. code-block:: bash

    poetry install
    pip install --no-deps quantum-metal

Developer Installation
----------------------

For developers who want to contribute to the project, install with additional development dependencies:

.. code-block:: bash

    poetry install --with docs,analysis
    pip install --no-deps quantum-metal
    pre-commit install

This will install:

- All project dependencies
- Documentation tools
- Analysis and testing tools
- Pre-commit hooks for code quality

Building Documentation
----------------------

To build the documentation yourself:

1. Install `pandoc <https://pandoc.org/>`_ (you may need to restart your computer to correctly set the path variables).
2. Run the following command:

.. code-block:: bash

    poetry run sphinx-build -b html docs/source docs/_build/html

Troubleshooting
===============

**Common Issues:**

- **Python Version Mismatch**: Ensure you're using Python 3.10 or 3.11 or 3.12
- **Dependency Conflicts**: If you encounter dependency conflicts, try installing in a fresh virtual environment
- **Ansys Connection Issues**: Make sure Ansys HFSS is correctly installed and licensed

Migration from qiskit-metal to quantum-metal
==============================================

If you have an existing QDesignOptimizer installation with qiskit-metal, follow these steps to migrate:

**Delete your existing environment and recreate it** to ensure all dependencies are correctly installed:

.. code-block:: bash

    # Deactivate and remove the old environment
    # If using conda:
    conda deactivate
    conda env remove -n qdesignenv

    # If using venv:
    deactivate
    rm -rf qdesignenv  # On Windows: rmdir /s qdesignenv

Then follow the installation instructions above to create a fresh environment.

**Note:** quantum-metal v0.5+ uses PySide6 instead of PySide2. The import paths remain ``qiskit_metal`` for backward compatibility.

For more help, please open an issue on the `GitHub repository <https://github.com/202Q-lab/QDesignOptimizer/issues>`_.
