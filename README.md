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
<!-- 
## :boom: Sample Usage
List available imputers
```python
from hyperimpute.plugins.imputers import Imputers

imputers = Imputers()

imputers.list()
```
Impute a dataset using one of the available methods
```python
import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]])

method = "gain"

plugin = Imputers().get(method)
out = plugin.fit_transform(X.copy())

print(method, out)
```
Specify the baseline models for HyperImpute
```python
import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]])

plugin = Imputers().get(
    "hyperimpute",
    optimizer="hyperband",
    classifier_seed=["logistic_regression"],
    regression_seed=["linear_regression"],
)

out = plugin.fit_transform(X.copy())
print(out)
```
Use an imputer with a SKLearn pipeline
```python
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from hyperimpute.plugins.imputers import Imputers

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]])
y = pd.Series([1, 2, 1, 2])

imputer = Imputers().get("hyperimpute")

estimator = Pipeline(
    [
        ("imputer", imputer),
        ("forest", RandomForestRegressor(random_state=0, n_estimators=100)),
    ]
)

estimator.fit(X, y)
``` -->