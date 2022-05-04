# Automated generation of conditional moment equations for stochastic reaction networks
This repository contains Python code to automatically generate and save conditional moment equations for a set of chemical species in a stochastic reaction network. This tool is the basis for **paper**. *(Last update: April 2022.)*

## Content
- `ConditionalMomentGenerator.py` contains the source code with all functions that are needed to generate the moment equations.
- `HOW TO generate conditional moment equations.ipynb` is a tutorial explaining how to use the generator tool.
- `SSA.py`, `snSSA.py`, and `plotting.py` are further scripts containing code to reproduce the case study of **paper**.

For further information, contact Hanna Wiederanders (wiederan@mpi-cbg.de).

## Installation
The provided code was written using Python v3.7.9 and uses the following libraries:
- [Numpy v1.19.3](https://www.numpy.org/)
- [Sympy v1.7.1](https://www.sympy.org/)
- [Dill v0.3.4](https://pypi.org/project/dill/)
- [Matplotlib v3.3.3](https://matplotlib.org/)

In case you do not have anything installed, the easiest way to start is to install using Anaconda. Instructions to install Python and Jupyter can be found at https://jupyter.readthedocs.io/en/latest/install.html. Numpy, Sympy, and Matplotlib are already included in the Anaconda installation. To install Dill, use the package manager [pip](https://pip.pypa.io/en/stable/):
```bash
pip install dill
```

## Usage
The `ConditionalMomentGenerator.py` python script contains functions to generate conditional moment equations as described in **paper**. After installation of the necessary packages, you can download the script and use the functions as explained in the Jupyter notebook `HOW TO...ipynb`.