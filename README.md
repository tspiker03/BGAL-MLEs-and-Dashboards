# BGAL Distribution Package

This package provides functionality for the Bivariate Asymmetric Laplace (BGAL) distribution.

## Installation

```bash
pip install .
```

Alternatively, you can include the `bgal_distribution` directory in your project.

## Usage

```python
import numpy as np
from bgal_distribution import BGALDistribution

# Sample data (replace with your actual data)
X_data = np.random.gamma(shape=2, scale=2, size=100)
y_data = np.random.normal(loc=0, scale=1, size=100)

# Initialize and fit the distribution
bgal_dist = BGALDistribution()
bgal_dist.fit(X_data, y_data)

# Get MLE parameters
mle_params = bgal_dist.mle_params()
print("MLE Parameters:", mle_params)

# Generate random samples
y_samples = bgal_dist.rvs(size=100, X_data=X_data)
print("Random Samples:", y_samples[:5])

# Perform goodness-of-fit test
gof_results = bgal_dist.goodness_of_fit(y_data, X_data=X_data)
print("\nGoodness-of-Fit Test (Anderson-Darling):")
print("Statistic:", gof_results["statistic"])
print("P-value:", gof_results["p_value"])
print("Null Hypothesis:", gof_results["null_hypothesis"])
print("Interpretation:", gof_results["interpretation"])
```

## Package Files

- `bgal_distribution/distribution.py`: Contains the `BGALDistribution` class, which includes methods for fitting the distribution, getting MLE parameters, generating random samples, and performing goodness-of-fit tests.
- `bgal_distribution/mle.py`: Contains functions for calculating Maximum Likelihood Estimates (MLEs) for the BGAL distribution parameters (delta, r, mu).
- `bgal_distribution/gof.py`: Contains functions for goodness-of-fit tests, currently including the Anderson-Darling test.
- `bgal_distribution/__init__.py`:  Initializes the package and makes the classes and functions accessible.
- `README.md`: This file, providing an overview of the package and usage instructions.
- `setup.py`:  Installation script for the package.

## Goodness-of-Fit Test

The package includes the Anderson-Darling test for goodness of fit.

**Null Hypothesis:** The data follows a BGAL distribution.

The interpretation of the test is based on the p-value. If the p-value is greater than 0.05, we fail to reject the null hypothesis, suggesting that the data may follow a BGAL distribution. Otherwise, we reject the null hypothesis.

## Dependencies

- numpy
- scipy

---
This README provides a basic guide to using the BGAL distribution package. For more details, refer to the code and docstrings within the package files.
