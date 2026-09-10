import unittest
import numpy as np
from scripts.run_review_revision_robustness_protocol import seed_block_bootstrap, _rate_configs


class SeedBlockBootstrapTests(unittest.TestCase):
    def test_identical_copies_do_not_artificially_narrow_interval(self):
        row = np.arange(10, dtype=float)[None, :]
        np.testing.assert_array_equal(
            seed_block_bootstrap(row), seed_block_bootstrap(np.repeat(row, 3, axis=0))
        )

    def test_anticorrelated_copies_preserve_cancellation(self):
        row = np.arange(10, dtype=float)
        np.testing.assert_array_equal(
            seed_block_bootstrap(np.stack([row, -row])), np.zeros(10000)
        )

    def test_complete_convergence_coverage(self):
        self.assertEqual(sum(len(_rate_configs(x)) for x in ("ml1m", "amazon_kindle", "netflix5k5k")), 8)

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            seed_block_bootstrap(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
