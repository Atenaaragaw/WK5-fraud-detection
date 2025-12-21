import unittest
import pandas as pd
import numpy as np
import sys
import os

# Allow the test to find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import map_ip_to_country, engineer_features

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a tiny fake fraud dataset
        self.sample_fraud = pd.DataFrame({
            'user_id': [1, 2],
            'signup_time': ['2025-01-01 10:00:00', '2025-01-01 10:00:00'],
            'purchase_time': ['2025-01-01 10:00:10', '2025-01-01 11:00:00'],
            'ip_address': [150, 450],
            'device_id': ['A', 'B']
        })

        # Create a tiny fake IP mapping dataset
        self.sample_ip = pd.DataFrame({
            'lower_bound_ip_address': [100, 400],
            'upper_bound_ip_address': [200, 500],
            'country': ['CountryA', 'CountryB']
        })

    def test_ip_mapping(self):
        # Test if IP 150 maps to CountryA
        result = map_ip_to_country(self.sample_fraud, self.sample_ip)
        country = result.loc[result['user_id'] == 1, 'country'].values[0]
        self.assertEqual(country, 'CountryA')

    def test_time_features(self):
        # Test if time_since_signup calculation is correct
        result = engineer_features(self.sample_fraud)
        # 10 seconds difference for user 1
        self.assertEqual(result.loc[result['user_id'] == 1, 'time_since_signup'].values[0], 10.0)

if __name__ == '__main__':
    unittest.main()