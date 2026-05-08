# DDSM (Data-driven Dynamical Systems Modeling) <!-- omit in toc -->

[![Version](https://img.shields.io/badge/version-v0.1.1-1a7f37)](https://github.com/fumito100111/ddsm/releases)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/fumito100111/ddsm)](https://github.com/fumito100111/ddsm/issues)
[![GitHub stars](https://img.shields.io/github/stars/fumito100111/ddsm?style=social)](https://github.com/fumito100111/ddsm/stargazers)

## Overview <!-- omit in toc -->

`DDSM` is a Python library for Data-driven Dynamical systems modeling.
This library provides tools for modeling and analyzing dynamical systems using data-driven approaches.

> [!NOTE]
> Estimators in `DDSM` are **fully compatible with the Scikit-learn API**.

## Table of Contents <!-- omit in toc -->

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Supported Estimators](#supported-estimators)
    - [EDMD](#edmd)
    - [gEDMD](#gedmd)
    - [SINDy](#sindy)
  - [Utilities](#utilities)
    - [ddsm.utils.generator\_to\_eq](#ddsmutilsgenerator_to_eq)
- [References](#references)
- [Open Source Software](#open-source-software)
- [License](#license)

## Requirements

`DDSM` requires Python 3.12 or later.

## Installation

You can install the latest version of `DDSM` using pip (From GitHub):

```bash
python -m pip install git+https://github.com/fumito100111/ddsm.git@v0.1.1
```

> [!TIP]
> If you want to install a specific version, specify it after the `@` symbol.
> For example, to install version `v0.1.1`, use `@v0.1.1`.

## Usage

### Supported Estimators

- [EDMD](#edmd)
- [gEDMD](#gedmd)
- [SINDy](#sindy)

> [!NOTE]
> For more details on how to use each estimator, please refer to the [samples](./samples) directory.

#### EDMD

```python
import numpy as np
from ddsm.dicts import MonomialsDict
from ddsm.estimators import EDMD

dt = 1.0e-3                 # Time interval between data points
x = np.random.rand(100, 2)  # Sample data at time t
y = np.sin(x)               # Sample target data at time t + dt

estimator = EDMD(
    psix_cls=MonomialsDict,
    psix_kwargs={"degree": 2},
    psiy_cls=MonomialsDict,
    psiy_kwargs={"degree": 2},
    reg='none'
)
estimator.fit(x, y)

K = estimator.right_K
L = estimator.left_L(dt=dt)
```

#### gEDMD

```python
import numpy as np
from ddsm.dicts import MonomialsDict
from ddsm.estimators import gEDMD

dt = 1.0e-3                 # Time interval between data points
x = np.random.rand(100, 2)  # Sample data at time t
y = np.sin(x)               # Sample target data at time t + dt
dx = (y - x) / dt           # Sample derivative data at time t

estimator = gEDMD(
    psi_cls=MonomialsDict,
    psi_kwargs={"degree": 2},
    reg='lasso',
    reg_kwargs={"alpha": 0.1}
)
estimator.fit(x, dx)

L = estimator.right_L
K = estimator.left_K(dt=dt)
```

#### SINDy

```python
import numpy as np
from ddsm.dicts import MonomialsDict
from ddsm.estimators import SINDy

dt = 1.0e-3                 # Time interval between data points
x = np.random.rand(100, 2)  # Sample data at time t
y = np.sin(x)               # Sample target data at time t + dt
dx = (y - x) / dt           # Sample derivative data at time t

estimator = SINDy(
    psi_cls=MonomialsDict,
    psi_kwargs={"degree": 2},
    threshold=0.1,
    max_iter=20
)
estimator.fit(x, dx)

L = estimator.right_L
```

### Utilities

`DDSM` also provides the following utility functions:

- [`ddsm.utils.generator_to_eq`](#ddsmutilsgenerator_to_eq): Convert a generator matrix to a system of equations.

> [!NOTE]
> For more details on how to use the utility functions, please refer to the [samples](./samples) directory.

#### ddsm.utils.generator_to_eq

This function converts a generator matrix obtained into a system of equations, separating the drift and diffusion terms.

```python
import numpy as np
from ddsm.dicts import MonomialsDict
from ddsm.estimators import gEDMD
from ddsm.utils import generator_to_eq

dt = 1.0e-3                 # Time interval between data points
x = np.random.rand(100, 2)  # Sample data at time t
y = np.sin(x)               # Sample target data at time t + dt
dx = (y - x) / dt           # Sample derivative data at time t

estimator = gEDMD(
    psi_cls=MonomialsDict,
    psi_kwargs={"degree": 2},
    reg='lasso',
    reg_kwargs={"alpha": 0.1}
)
estimator.fit(x, dx)

right_L = estimator.right_L
psi = estimator.psi_
eq = generator_to_eq(right_L, psi)
print('----- Drift Term -----')
print(eq.drift)
print('----- Diffusion Term -----')
print(eq.diff)
```

## References

1. Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition. *Journal of Nonlinear Science*, 25, 1307-1346.

2. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences (PNAS)*.

3. Klus, S., Nüske, F., Peitz, S., Niemann, J. H., Clementi, C., & Schütte, C. (2020). Data-driven approximation of the Koopman generator: Model reduction, system identification, and control. *Physica D: Nonlinear Phenomena*, 406, 132416.

## Open Source Software

`DDSM` uses the following open source software:

- [NumPy](https://numpy.org/)
- [SymPy](https://www.sympy.org/)
- [SciPy](https://scipy.org/)
- [Scikit-learn](https://scikit-learn.org/)

## License

This project is licensed under the `MIT License`.
See the [`LICENSE`](LICENSE) file for details.