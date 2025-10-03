# QDesignOptimizer

![docs](https://github.com/202Q-lab/QDesignOptimizer/actions/workflows/deploy_docs.yml/badge.svg)
![Lint-Pytest-Mypy](https://github.com/202Q-lab/QDesignOptimizer/actions/workflows/analysis.yml/badge.svg)

QDesignOptimizer (QDO) is a Python package which optimizes the design of quantum devices. It integrates with the Qiskit Metal framework and uses HFSS simulations to iteratively improve superconducting qubit designs.

## Documentation

For detailed documentation, visit [https://202Q-lab.github.io/QDesignOptimizer/](https://202Q-lab.github.io/QDesignOptimizer/)

## Installation

### Requirements

- Python 3.10
- Ansys Electronics Desktop 2021 R2

#### For Windows
- VSCode (recommended)
- install https://visualstudio.microsoft.com/visual-cpp-build-tools/, make sure to select Build tools with C++

### Installation with pip

Install the package via pip:

```bash
pip install qdesignoptimizer
pip install --no-deps qiskit-metal==0.1.5
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
pip install --no-deps qiskit-metal==0.1.5
```

#### Developer Installation

For developers who want to contribute:

```bash
poetry install --with docs,analysis
pip install --no-deps qiskit-metal==0.1.5
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

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/202Q-lab/QDesignOptimizer/blob/main/LICENSE.txt) file for details.
