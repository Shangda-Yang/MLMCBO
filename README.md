# MLMCBO

MLMCBO is a library for Multilevel Monte Carlo Bayesian Optimization based
on [BoTorch](https://github.com/pytorch/botorch/tree/main).

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

File ```tests.py``` in the ```tutorials``` folder demonstrates a basic usage of the MLMC q-Expected Improvement (qEI)
acquisition functions:

- MC One-Step Lookahead EI
- MC One-Step Lookahead qEI
- MLMC One-Step Lookahead qEI
- MLMC Two-Step Lookahead qEI

## License
BoTorch is MIT licensed, as found in the [LICENSE](LICENSE) file.
MLMCBO is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.