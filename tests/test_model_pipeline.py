from pathlib import Path
import json
import tempfile
import unittest
import torch
from drbridge.model import ClinicalTokenTransformer, ModelConfig
from drbridge.adjustment import AdjustmentConfig
from drbridge.pipeline import frozen_refinement, run_joint_monitor, correction_outcomes
from drbridge.integrity import state_dict_sha256, file_sha256, load_checkpoint
from fixtures import FEATURE_NAMES, make_model, sample_inputs


class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shapes_and_directional_normalization(self):
        model = make_model()
        image, clinical, *_ = sample_inputs()
        with torch.no_grad():
            encoded = model.encode(image, clinical)
            out = model.infer_from_tokens(encoded["image_tokens"], encoded["clinical_tokens"])
        self.assertEqual(encoded["image_tokens"].shape, (3, 256, 32))
        self.assertEqual(encoded["clinical_tokens"].shape, (3, 26, 32))
        self.assertEqual(out["attention_c2i"].shape, (3, 4, 26, 256))
        self.assertEqual(out["attention_i2c"].shape, (3, 4, 256, 26))
        self.assertTrue(torch.allclose(out["attention_c2i"].sum(-1), torch.ones(3, 4, 26), atol=1e-6))
        self.assertTrue(torch.allclose(out["attention_i2c"].sum(-1), torch.ones(3, 4, 256), atol=1e-6))
        self.assertEqual(out["logits"].shape, (3, 5))

    def test_gradients_reach_both_attention_directions(self):
        model = make_model()
        image, clinical, *_ = sample_inputs()
        logits = model(image, clinical)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
        for attention in (model.clinical_to_image_attn, model.image_to_clinical_attn):
            self.assertGreater(attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_missingness_embedding(self):
        model = make_model()
        values = torch.zeros(1, 26)
        values[0, 0] = float("nan")
        cls, tokens = model.clinical_encoder(values)
        self.assertTrue(torch.isfinite(cls).all())
        self.assertTrue(torch.isfinite(tokens).all())
        with self.assertRaises(ValueError):
            model.clinical_encoder(values, torch.zeros(1, 26, dtype=torch.bool))

    def test_frozen_refinement_changes_attention_not_weights(self):
        model = make_model()
        image, clinical, p, present, boxes = sample_inputs()
        before = state_dict_sha256(model)
        out = frozen_refinement(model, image, clinical, p, present, boxes,
                                 FEATURE_NAMES, AdjustmentConfig(.4, .8))
        self.assertEqual(before, out["state_sha256_after"])
        self.assertTrue(torch.equal(out["baseline"]["logits"][1:], out["refined"]["logits"][1:]))
        self.assertFalse(torch.equal(out["baseline"]["attention_c2i"][0], out["refined"]["attention_c2i"][0]))
        self.assertFalse(torch.equal(out["baseline"]["logits"][0], out["refined"]["logits"][0]))
        self.assertTrue(torch.equal(out["baseline"]["attention_i2c"], out["refined"]["attention_i2c"]))

    def test_association_anomaly_does_not_block_refinement(self):
        _, _, p, present, boxes = sample_inputs()
        a = torch.full((3, 2, 26, 256), 1/256)
        a[:, :, :6, :2] = .5
        a[:, :, :6, 2:] = 0
        out = run_joint_monitor(a, p, present, boxes, FEATURE_NAMES, AdjustmentConfig(.4, .8))
        self.assertTrue(out["indicator"]["association_anomaly"][0].item())
        self.assertTrue(out["control"]["automatic_eligible"][0].item())
        self.assertTrue(out["specialist_review"][0].item())
        self.assertFalse(out["control"]["automatic_eligible"][1].item())
        self.assertTrue(out["specialist_review"][1].item())

    def test_original_training_flags_restored(self):
        model = make_model().train()
        model.clinical_encoder.eval()
        modes = [m.training for m in model.modules()]
        image, clinical, p, present, boxes = sample_inputs()
        frozen_refinement(model, image, clinical, p, present, boxes, FEATURE_NAMES, AdjustmentConfig(.4, .8))
        self.assertEqual(modes, [m.training for m in model.modules()])

    def test_outcomes_form_disjoint_partition(self):
        y = torch.tensor([0, 0, 0, 0, 0, 1])
        initial = torch.tensor([1, 2, 1, 0, 1, 1])
        refined = torch.tensor([0, 1, 1, 1, 2, 1])
        out = correction_outcomes(initial, refined, y)
        self.assertEqual(out["complete"].sum().item(), 1)
        self.assertEqual(out["partial"].sum().item(), 1)
        self.assertEqual(out["unchanged"].sum().item(), 2)
        self.assertEqual(out["correct_to_incorrect"].sum().item(), 1)
        self.assertEqual(out["incorrect_to_incorrect"].sum().item(), 1)
        self.assertTrue((sum(out[x].int() for x in ["complete", "partial", "unchanged", "nonimproving"]) == 1).all())

    def test_checkpoint_hash_and_roundtrip(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"weights.pt"
            torch.save(model.state_dict(), path)
            digest = file_sha256(path)
            self.assertEqual(load_checkpoint(make_model(), path, expected_sha256=digest), state_dict_sha256(model))
            with self.assertRaises(ValueError):
                load_checkpoint(model, path, expected_sha256="0"*64)


if __name__ == "__main__":
    unittest.main()
