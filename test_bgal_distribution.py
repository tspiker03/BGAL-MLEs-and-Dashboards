import unittest
import pandas as pd
import numpy as np
from bgal_distribution import BGALDistribution

class TestBGALDistribution(unittest.TestCase):

    def setUp(self):
        self.bgal_dist = BGALDistribution()
        self.sample_data = pd.DataFrame({
            'X': np.random.rand(100),
            'Y': np.random.rand(100)
        })

    def test_fit_method(self):
        self.bgal_dist.fit(self.sample_data)
        self.assertIsNotNone(self.bgal_dist.get_mle_params(), "Fit method should calculate MLE parameters.")

    def test_get_mle_params(self):
        self.bgal_dist.fit(self.sample_data)
        params = self.bgal_dist.get_mle_params()
        self.assertIsInstance(params, dict, "get_mle_params should return a dictionary.")
        self.assertIn('delta', params)
        self.assertIn('r', params) # sigma in theory
        self.assertIn('mu', params)

    def test_goodness_of_fit_tests(self):
        self.bgal_dist.fit(self.sample_data)
        gof_results = self.bgal_dist.goodness_of_fit_tests(self.sample_data)
        self.assertIsInstance(gof_results, dict, "goodness_of_fit_tests should return a dictionary.")
        self.assertIn('kolmogorov_smirnov', gof_results)
        self.assertIn('anderson_darling', gof_results)
        self.assertIn('chi_squared', gof_results)

        self.assertIn('null_hypothesis', gof_results['kolmogorov_smirnov'])
        self.assertIn('p_value', gof_results['kolmogorov_smirnov'])
        self.assertIn('statistic', gof_results['kolmogorov_smirnov'])

        self.assertIn('null_hypothesis', gof_results['anderson_darling'])
        self.assertIn('critical_values', gof_results['anderson_darling'])
        self.assertIn('statistic', gof_results['anderson_darling'])
        self.assertIn('significance_level', gof_results['anderson_darling'])

        self.assertIn('null_hypothesis', gof_results['chi_squared'])
        self.assertIn('p_value', gof_results['chi_squared'])
        self.assertIn('statistic', gof_results['chi_squared'])


if __name__ == '__main__':
    unittest.main()
