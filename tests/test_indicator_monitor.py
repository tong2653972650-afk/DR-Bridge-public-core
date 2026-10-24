import unittest
import torch
from drbridge.indicator_monitor import verify_indicator_associations
from fixtures import FEATURE_NAMES


class IndicatorMonitorTests(unittest.TestCase):
    def attention(self):
        a = torch.full((1, 2, 26, 256), 1/256)
        a[:, :, :6, :64] = 1/64
        a[:, :, :6, 64:] = 0
        return a

    def test_coverage_weighted_scores(self):
        a = self.attention()
        coverage = torch.zeros(1, 1, 256)
        coverage[..., :64] = .5
        out = verify_indicator_associations(a, coverage, torch.ones(1, 1, dtype=torch.bool), FEATURE_NAMES)
        self.assertTrue(torch.allclose(out["scores"][0, 0, :6], torch.full((6,), .5)))
        self.assertAlmostEqual(out["scores"][0, 0, 6].item(), .125)
        self.assertTrue(out["association_anomaly"].item())

    def test_top6_zero_hit_and_stable_ties(self):
        coverage = torch.zeros(1, 1, 256)
        coverage[..., :64] = 1
        out = verify_indicator_associations(self.attention(), coverage, torch.ones(1, 1, dtype=torch.bool), FEATURE_NAMES)
        self.assertEqual(out["top6_indices"][0, 0].tolist(), list(range(6)))
        self.assertEqual(out["reference_hits"].item(), 0)

    def test_any_reference_hit_passes(self):
        a = self.attention()
        a[:, :, 10] = a[:, :, 0]
        a[:, :, 5] = 1/256
        coverage = torch.zeros(1, 1, 256)
        coverage[..., :64] = 1
        out = verify_indicator_associations(a, coverage, torch.ones(1, 1, dtype=torch.bool), FEATURE_NAMES)
        self.assertFalse(out["association_anomaly"].item())
        self.assertEqual(out["reference_hits"].item(), 1)

    def test_query_identity_follows_input_order(self):
        names = list(FEATURE_NAMES)
        names[0], names[10] = names[10], names[0]
        coverage = torch.zeros(1, 1, 256)
        coverage[..., :64] = 1
        out = verify_indicator_associations(self.attention(), coverage, torch.ones(1, 1, dtype=torch.bool), names)
        self.assertFalse(out["association_anomaly"].item())

    def test_unverified_location_does_not_flag(self):
        out = verify_indicator_associations(self.attention(), torch.zeros(1, 1, 256),
                                             torch.zeros(1, 1, dtype=torch.bool), FEATURE_NAMES)
        self.assertFalse(out["association_anomaly"].item())
        self.assertFalse(out["has_verified_location"].item())

    def test_missing_reference_schema_rejected(self):
        with self.assertRaises(ValueError):
            verify_indicator_associations(self.attention(), torch.ones(1, 1, 256),
                                           torch.ones(1, 1, dtype=torch.bool), [f"x{i}" for i in range(26)])

    def test_empty_verified_region_rejected(self):
        with self.assertRaises(ValueError):
            verify_indicator_associations(self.attention(), torch.zeros(1, 1, 256),
                                           torch.ones(1, 1, dtype=torch.bool), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
