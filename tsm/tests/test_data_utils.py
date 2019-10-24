from unittest import TestCase
import pandas as pd
import numpy as np
from tsm.data_utils import compress_memory_usage


class TestDataUtils(TestCase):

    def setUp(self):
        np.random.seed(10)
        self.test_df = pd.DataFrame({
            'A': np.random.randint(100, size=10),
            'B': np.random.random(size=10),
            'C': 50 - np.random.randint(100, size=10),
            'D': 0.5 - np.random.random(size=10),
            'E': np.random.randint(int(1e10), size=10),
            'F': ['name' for _ in range(10)],
            'G': [1, 2, 3, 4, 5, 6, 7, 8, None, 10]
        })

    def test_compress_memory_usage(self):
        test_case, _ = compress_memory_usage(self.test_df)
        self.assertEqual(test_case.dtypes['A'], 'uint8')
        self.assertEqual(test_case.dtypes['B'], 'float32')
        self.assertEqual(test_case.dtypes['C'], 'int8')
        self.assertEqual(test_case.dtypes['D'], 'float32')
        self.assertEqual(test_case.dtypes['E'], 'uint64')
        self.assertEqual(test_case.dtypes['F'], 'object')
        self.assertEqual(test_case.dtypes['G'], 'uint8')
        self.assertEqual(test_case.G.isna().sum(), 0)
        self.assertIn(self.test_df.G.min() - 1, test_case.G)
