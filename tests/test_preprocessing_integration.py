import unittest
import torch
from PIL import Image
from drbridge.preprocessing import prepare_evidence_image
from drbridge.image_monitor import ImageEvidenceMonitor
from drbridge.adjustment import AdjustmentConfig
from drbridge.pipeline import infer_with_joint_monitor, frozen_refinement
from fixtures import FEATURE_NAMES, FixtureEvidenceEncoder, make_model, sample_inputs


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_padding_resize_and_normalization(self):
        im = Image.new("RGB", (40, 20), (255, 255, 255))
        x = prepare_evidence_image(im)
        self.assertEqual(x.shape, (3, 224, 224))
        self.assertAlmostEqual(x[0, 0, 100].item(), -.485/.229, places=5)
        self.assertAlmostEqual(x[0, 112, 112].item(), (1-.485)/.229, places=5)

    def test_grayscale_is_converted_to_rgb(self):
        self.assertEqual(prepare_evidence_image(Image.new("L", (20, 20))).shape, (3, 224, 224))

    def test_full_joint_entry_runs_all_components(self):
        model = make_model()
        monitor = ImageEvidenceMonitor(encoder=FixtureEvidenceEncoder(), feature_dim=3)
        with torch.no_grad():
            monitor.head.weight.zero_()
            monitor.head.bias.copy_(torch.tensor([0., 5., 0., 0.]))
        image, clinical, _, present, boxes = sample_inputs()
        out = infer_with_joint_monitor(model, monitor, image, clinical,
                                        torch.randn(3, 2, 3, 224, 224), present, boxes,
                                        FEATURE_NAMES, AdjustmentConfig(.4, .8))
        self.assertTrue(out["joint"]["control"]["automatic_eligible"].all())
        self.assertEqual(out["refined_probabilities"].shape, (3, 5))
        self.assertEqual(out["state_sha256_before"], out["state_sha256_after"])

    def test_inputs_are_not_modified(self):
        model = make_model()
        image, clinical, p, present, boxes = sample_inputs()
        inputs = [image, clinical, p, present, boxes]
        copies = [x.clone() for x in inputs]
        frozen_refinement(model, image, clinical, p, present, boxes, FEATURE_NAMES, AdjustmentConfig(.4, .8))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, copies)))


if __name__ == "__main__":
    unittest.main()

