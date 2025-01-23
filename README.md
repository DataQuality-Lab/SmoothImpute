# SmoothImpute - A comprehensive imputation library

<div align="center">

[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zGm4VeXsJ-0x6A5_icnknE7mbJ0knUig?usp=sharing)
[![Tutorials](https://github.com/DataQuality-lab/SmoothImpute/actions/workflows/test_tutorials.yml/badge.svg)](TBD)
[![Documentation Status](https://readthedocs.org/projects/smoothimpute/badge/?version=latest)](https://smoothimpute.readthedocs.io/en/latest/?badge=latest)


[![arXiv](https://img.shields.io/badge/arXiv-2206.07769-b31b1b.svg)](https://arxiv.org/abs/2206.07769)
[![](https://pepy.tech/badge/smoothimpute)](https://pypi.org/project/smoothimpute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

![image](https://github.com/DataQuality-lab/SmoothImpute/raw/main/docs/arch.png "SmoothImpute")

</div>


SmoothImpute simplifies the selection process of a data imputation algorithm for your ML pipelines.
It includes various novel algorithms for missing data and is compatible with [sklearn](https://scikit-learn.org/stable/).


## SmoothImpute features
- :rocket: Fast and extensible dataset imputation algorithms, compatible with sklearn.
- :key: New iterative imputation method: SmoothImpute.
- :cyclone: Classic methods: MICE, MissForest, GAIN, MIRACLE, MIWAE, Sinkhorn, SoftImpute, etc.
- :fire: Pluginable architecture.

## :rocket: Installation

The library can be installed from PyPI using
```bash
$ pip install smoothimpute
```
or from source, using
```bash
$ pip install .
```
