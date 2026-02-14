# QDesignOptimizer

![docs](https://github.com/202Q-lab/QDesignOptimizer/actions/workflows/deploy_docs.yml/badge.svg)
![Lint-Pytest-Mypy](https://github.com/202Q-lab/QDesignOptimizer/actions/workflows/analysis.yml/badge.svg)

QDesignOptimizer (QDO) is a Python package which optimizes the design of quantum devices. It integrates with the Qiskit Metal framework and uses HFSS simulations to iteratively improve superconducting qubit designs.

## Documentation

For detailed documentation, visit [https://202Q-lab.github.io/QDesignOptimizer/](https://202Q-lab.github.io/QDesignOptimizer/)

## Installation

### Requirements

- Python 3.10 or 3.11
- Ansys Electronics Desktop 2021 R2

#### For Windows
- VSCode (recommended)
- install https://visualstudio.microsoft.com/visual-cpp-build-tools/, make sure to select Build tools with C++

### Installation with pip

Install the package via pip:

```bash
pip install qdesignoptimizer
pip install --no-deps quantum-metal
```

### Installation from GitHub repository

#### Clone the Repository

```bash
git clone https://github.com/202Q-lab/QDesignOptimizer
cd QDesignOptimizer
```

#### Create a Virtual Environment

It's strongly recommended to install in a separate virtual environment with Python 3.10.

**Using Conda:**

```bash
conda env create -f environment.yml
conda activate qdesignenv
```

**Using venv:**

```bash
# Create new virtual environment
python3.10 -m venv qdesignenv

# Activate the environment
# On Linux/MacOS:
source qdesignenv/bin/activate
# On Windows:
qdesignenv\Scripts\activate

# Verify Python version
python --version

# Install poetry if not already available
pip install poetry
```

#### User Installation

For regular users:

```bash
poetry install
pip install --no-deps quantum-metal
```

#### Developer Installation

For developers who want to contribute:

```bash
poetry install --with docs,analysis
pip install --no-deps quantum-metal
pre-commit install
```

This installs:
- All project dependencies
- Documentation tools
- Analysis and testing tools
- Pre-commit hooks for code quality

To build the documentation yourself, install [pandoc](https://pandoc.org/) and run:

```bash
poetry run sphinx-build -b html docs/source docs/build
```

## Migration from qiskit-metal to quantum-metal

If you have an existing QDesignOptimizer installation with qiskit-metal, follow these steps to migrate:

1. **Uninstall old packages:**
   ```bash
   pip uninstall -y qiskit-metal pyside2
   ```

2. **Update your environment:**
   ```bash
   poetry install --with docs,analysis
   pip install --no-deps quantum-metal
   ```

3. **Verify the installation:**
   ```python
   from qiskit_metal import designs, MetalGUI
   from qdesignoptimizer import DesignAnalysis
   ```

**Note:** quantum-metal v0.5+ uses PySide6 instead of PySide2. The import paths remain `qiskit_metal` for backward compatibility.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/202Q-lab/QDesignOptimizer/blob/main/LICENSE.txt) file for details.
