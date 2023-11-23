# MLMCBO

MLMCBO is a library for the Multilevel Monte Carlo Bayesian Optimization method [to be linked to arxiv paper]. The package is based on the package [BoTorch](https://github.com/pytorch/botorch/tree/main).

**TL;DR** Multilevel Monte Carlo accelerates nested Monte Carlo approximations (sometimes by several orders of magnitude). The current package leverages this technology to speed up the evaluation of lookahead acquisition functions for Bayesian optimization.


## Installation

**Installation Requirements**

- BoTorch == 0.9.2
- Python >= 3.9 (BoTorch requirement)
- NumPy >= 1.18

### Pre-installation

At this development stage it is recommended to create a separate conda environment to minimize possible conflicts.

```bash
conda create --name mlmcbo python=3.11
conda activate mlmcbo
```

### Option 1: Editable/dev install

```bash
git clone https://github.com/Shangda-Yang/MLMCBO.git
cd MLMCBO
pip install -e .
```

## Getting Started

File ```tests.py``` in the ```tutorials``` folder demonstrates basic usage of the MLMC q-Expected Improvement (qEI) acquisition functions:

- MC One-Step Lookahead EI
- MC One-Step Lookahead qEI
- MLMC One-Step Lookahead qEI
- MLMC Two-Step Lookahead qEI
- MLMC Two-Step Lookahead qEI (beta)
- 
File ```testWholeBo.py``` in the same folder demonstrates an example of a whole BO algorithm using MLMC for 2-EI and plots the results..
It optimizes a self-defined function defined in ```objectiveFunction.py``` in the ```utils``` folder.
=======

## License
MLMCBO is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
