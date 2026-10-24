import unittest
import torch
from drbridge.image_monitor import ImageEvidenceMonitor, summarize_image_evidence, classify_evidence_batch
from fixtures import FixtureEvidenceEncoder


class ImageMonitorTests(unittest.TestCase):
    def test_four_class_forward_and_learning(self):
        torch.manual_seed(12)
        model = ImageEvidenceMonitor(encoder=FixtureEvidenceEncoder(), feature_dim=3)
        colors = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])
        images = colors[:, :, None, None].expand(4, 3, 224, 224)
        labels = torch.arange(4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
        first = torch.nn.functional.cross_entropy(model(images)["logits"], labels).item()
        for _ in range(30):
            optimizer.zero_grad()
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output["logits"], labels)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first*0.5)
        self.assertEqual(output["probabilities"].shape, (4, 4))
        self.assertTrue(torch.allclose(output["probabilities"].sum(-1), torch.ones(4)))

    def test_top2_is_mean_of_two_largest(self):
        p = torch.tensor([[[.8, .1, .05, .05], [.1, .8, .05, .05], [.5, .2, .2, .1]]])
        summary = summarize_image_evidence(p, torch.ones(1, 3, dtype=torch.bool))
        self.assertAlmostEqual(summary["top2_score"].item(), .7, places=6)
        self.assertTrue(summary["image_anomaly"].item())

    def test_one_item_uses_one_probability(self):
        p = torch.tensor([[[.1, .7, .1, .1], [.1, .7, .1, .1]]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(summarize_image_evidence(p, mask)["top2_score"].item(), .9, places=6)

    def test_empty_evidence_has_no_fabricated_score(self):
        p = torch.empty(2, 0, 4)
        summary = summarize_image_evidence(p, torch.empty(2, 0, dtype=torch.bool))
        self.assertTrue(torch.isnan(summary["top2_score"]).all())
        self.assertFalse(summary["image_anomaly"].any())

    def test_other_indeterminate_blocks_automatic_path(self):
        p = torch.tensor([[[.01, .97, .01, .01], [.01, .01, .01, .97]]])
        s = summarize_image_evidence(p, torch.ones(1, 2, dtype=torch.bool))
        self.assertTrue(s["other_indeterminate"].item())
        self.assertFalse(s["automatic_eligible"].item())

    def test_padding_ignored(self):
        p = torch.tensor([[[.9, .05, .03, .02], [float("nan")]*4]])
        s = summarize_image_evidence(p, torch.tensor([[True, False]]))
        self.assertAlmostEqual(s["top2_score"].item(), .1, places=6)
        self.assertEqual(s["class_labels"][0, 1].item(), -1)

    def test_invalid_probability_vector_rejected(self):
        with self.assertRaises(ValueError):
            summarize_image_evidence(torch.ones(1, 1, 4), torch.ones(1, 1, dtype=torch.bool))

    def test_packed_network_inputs(self):
        monitor = ImageEvidenceMonitor(encoder=FixtureEvidenceEncoder(), feature_dim=3)
        p = classify_evidence_batch(monitor, torch.randn(2, 2, 3, 224, 224),
                                    torch.tensor([[True, False], [True, True]]))
        self.assertEqual(p.shape, (2, 2, 4))
        self.assertTrue(p[0, 1].eq(0).all())

    def test_network_input_size_checked(self):
        with self.assertRaises(ValueError):
            ImageEvidenceMonitor(encoder=FixtureEvidenceEncoder(), feature_dim=3)(torch.randn(1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()

