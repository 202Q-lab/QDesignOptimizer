# QDesignOptimizer

![docs](https://github.com/202Q-lab/QDesignOptimizer/actions/workflows/deploy_docs.yml/badge.svg)
![Lint-Pytest-Mypy](https://github.com/202Q-lab/QDesignOptimizer/actions/workflows/analysis.yml/badge.svg)

QDesignOptimizer (QDO) is a Python package which optimizes the design of quantum devices. It integrates with the Qiskit Metal framework and uses HFSS simulations to iteratively improve superconducting qubit designs. Developed by the [202Q-lab](https://202q-lab.se/) at Chalmers University of Technology.

## Documentation

For detailed documentation, visit [https://202Q-lab.github.io/QDesignOptimizer/](https://202Q-lab.github.io/QDesignOptimizer/)

## Videos

Short presentations and in-depth walkthroughs are available on YouTube:

- [APS March Meeting presentation (10 min)](https://www.youtube.com/watch?v=M940wW7Taos) — overview of QDO
- [Deep-dive Webinar (1 h)](https://www.youtube.com/watch?v=tdSur5loICM) — detailed technical walkthrough
- [Step-by-step intro Webinar (1 h)](https://www.youtube.com/watch?v=8chNX4i1Tk8) — guided introduction and demo

## Installation

### Quick Install

```bash
pip install qdesignoptimizer
pip install --no-deps quantum-metal
```

**Requirements:** Python 3.11 or 3.12, Ansys Electronics Desktop 2021 R2 or 2022 R2. 

For detailed installation instructions including virtual environment setup, developer installation, troubleshooting, and migration from qiskit-metal, see the [full installation guide](https://202Q-lab.github.io/QDesignOptimizer/installation.html).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/202Q-lab/QDesignOptimizer/blob/main/LICENSE.txt) file for details.
