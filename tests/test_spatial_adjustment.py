import unittest
import torch
from drbridge.spatial import box_coverage, pool_to_regions, expand_region_coefficients
from drbridge.adjustment import AdjustmentConfig, build_attention_control, reweight_attention
from fixtures import sample_inputs


class SpatialAdjustmentTests(unittest.TestCase):
    def test_cell_area_coverage(self):
        boxes = torch.tensor([[[0., 0., 32., 64.]]])
        w = box_coverage(boxes, torch.ones(1, 1, dtype=torch.bool), grid=(8, 8))
        self.assertAlmostEqual(w[0, 0, 0].item(), .5)
        self.assertAlmostEqual(w.sum().item(), .5)

    def test_outside_box_rejected(self):
        with self.assertRaises(ValueError):
            box_coverage(torch.tensor([[[600., 0., 700., 32.]]]), torch.ones(1, 1, dtype=torch.bool))

    def test_spatial_blocks_not_consecutive_flat_chunks(self):
        x = torch.zeros(1, 1, 26, 256)
        x[..., [0, 1, 16, 17]] = .25
        regional = pool_to_regions(x)
        self.assertTrue(regional[..., 0].eq(1).all())
        self.assertTrue(regional[..., 1:].eq(0).all())

    def test_expand_preserves_coarse_fine_correspondence(self):
        r = torch.ones(1, 64)
        r[0, 0] = 2
        out = expand_region_coefficients(r)
        self.assertEqual(torch.where(out[0] == 2)[0].tolist(), [0, 1, 16, 17])

    def test_suppression_and_enhancement_formula(self):
        _, _, p, mask, boxes = sample_inputs()
        c = build_attention_control(p, mask, boxes, AdjustmentConfig(.4, .8))
        self.assertAlmostEqual(c["regional_coefficients"][0, 0].item(), 1-.4*.97, places=6)
        self.assertAlmostEqual(c["regional_coefficients"][0, 18].item(), 1+.8*.94, places=6)
        self.assertTrue(c["regional_coefficients"][1:].eq(1).all())

    def test_empty_evidence_is_neutral(self):
        c = build_attention_control(torch.empty(1, 0, 4), torch.empty(1, 0, dtype=torch.bool),
                                    torch.empty(1, 0, 4), AdjustmentConfig(.4, .8))
        self.assertTrue(c["fine_coefficients"].eq(1).all())

    def test_mass_normalized_and_within_region_ratio_preserved(self):
        a = torch.arange(1., 257.).reshape(1, 1, 1, 256).expand(1, 2, 26, 256)
        a = a/a.sum(-1, keepdim=True)
        r = torch.ones(1, 64)
        r[0, 0] = .2
        c = expand_region_coefficients(r)
        out = reweight_attention(a, c)
        self.assertTrue(torch.allclose(out.sum(-1), torch.ones(1, 2, 26)))
        self.assertAlmostEqual((out[0, 0, 0, 0]/out[0, 0, 0, 16]).item(), 1/17, places=6)
        self.assertLess(out[0, 0, 0, 0].item(), a[0, 0, 0, 0].item())

    def test_joint_overlap_keeps_both_effects(self):
        p = torch.tensor([[[.1, .7, .1, .1], [.9, .05, .03, .02]]])
        boxes = torch.tensor([[[0., 0., 64., 64.], [0., 0., 64., 64.]]])
        c = build_attention_control(p, torch.ones(1, 2, dtype=torch.bool), boxes, AdjustmentConfig(.4, .8))
        self.assertAlmostEqual(c["regional_coefficients"][0, 0].item(), (1-.4*.9)*(1+.8*.9), places=6)

    def test_invalid_strength_rejected(self):
        for a, e in [(1., 1.), (-.1, 1.), (.5, -1.), (float("nan"), 1.)]:
            with self.assertRaises(ValueError):
                AdjustmentConfig(a, e)


if __name__ == "__main__":
    unittest.main()

