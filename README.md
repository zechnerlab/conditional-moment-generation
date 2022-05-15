# Automated generation of conditional moment equations for stochastic reaction networks
This repository contains Python code to automatically generate and save conditional moment equations for a set of chemical species in a stochastic reaction network. This tool is the basis for the paper "Automatic Generation of Conditional Moment Equations for Stochastic Reaction Networks" by HJ Wiederanders, AL Moor and C Zechner currently under submission. *(Last update: 9 May 2022.)*

## Content
- `ConditionalMomentGenerator.py` contains the source code with all functions that are needed to generate the moment equations.
- `HOW TO generate conditional moment equations.ipynb` is a tutorial explaining how to use the generator tool.
- The directory `case study` contains the files `case_study.py`, `SSA_multicore.py`, `snSSA.py`, and `plotting.py`. These are further scripts containing code to reproduce the case study equations, simulations, and plots of the paper.

For further information, contact Hanna Wiederanders (wiederan@mpi-cbg.de).

## Installation
The provided code was written using Python v3.7.9 and uses the following libraries:
- [Numpy v1.19.3](https://www.numpy.org/)
- [Sympy v1.7.1](https://www.sympy.org/)
- [Dill v0.3.4](https://pypi.org/project/dill/)
- [Matplotlib v3.3.3](https://matplotlib.org/)

In case you are new to Python, the simplest installation of Python, Jupyter notebooks and the required packages is by using [Anaconda](https://www.anaconda.com/products/distribution#windows), a package manager that contains all required packages, except for `Dill`. Instructions to install Python and Jupyter can be found at [here](https://jupyter.readthedocs.io/en/latest/install.html). To install Dill, run the following command in the terminal:
```bash
pip install dill
```

## Usage
The `ConditionalMomentGenerator.py` python script contains functions to generate conditional moment equations as described in **paper**. After installation of the necessary packages, you can download the script and use the functions as explained in the Jupyter notebook `HOW TO generate conditional moment equations.ipynb`.
