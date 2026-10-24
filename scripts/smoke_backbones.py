"""Offline forward checks with actual ConvNeXt-Small and ConvNeXt-Tiny."""

from pathlib import Path
import argparse
import json
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from drbridge.model import DRBridgeModel, ModelConfig
from drbridge.image_monitor import ImageEvidenceMonitor
from drbridge.adjustment import AdjustmentConfig
from drbridge.pipeline import infer_with_joint_monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    # Public geometry; unreported wrapper settings below are only test settings.
    config = ModelConfig(backbone="convnext_small", token_dim=256, num_heads=8,
                         clinical_layers=2, fusion_layers=1, dropout=0.0, head_hidden_dim=32)
    model = DRBridgeModel(config).to(args.device).eval()
    image = torch.randn(1, 3, 512, 512, device=args.device)
    clinical = torch.randn(1, 26, device=args.device)
    with torch.no_grad():
        _, out = model(image, clinical, return_attention=True)
    report = {"purpose": "untrained backbone forward verification", "device": args.device,
              "primary_logits_shape": list(out["logits"].shape),
              "clinical_to_image_shape": list(out["attention_c2i"].shape),
              "image_to_clinical_shape": list(out["attention_i2c"].shape),
              "primary_finite": bool(torch.isfinite(out["logits"]).all())}
    monitor = ImageEvidenceMonitor().to(args.device).eval()
    evidence = torch.randn(1, 1, 3, 224, 224, device=args.device)
    with torch.no_grad():
        out = monitor(evidence[:, 0])
    report.update({"evidence_logits_shape": list(out["logits"].shape),
                    "evidence_probabilities_sum": float(out["probabilities"].sum()),
                    "evidence_finite": bool(torch.isfinite(out["logits"]).all())})
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root/"config/clinical_schema.example.json").read_text(encoding="utf-8"))["feature_names"]
    present = torch.ones(1, 1, dtype=torch.bool, device=args.device)
    boxes = torch.tensor([[[0., 0., 64., 64.]]], device=args.device)
    result = infer_with_joint_monitor(model, monitor, image, clinical, evidence,
                                      present, boxes, names, AdjustmentConfig(.4, .8))
    report.update({"joint_pipeline_finite": bool(torch.isfinite(result["refined"]["logits"]).all()),
                   "joint_pipeline_weights_unchanged": result["state_sha256_before"] == result["state_sha256_after"],
                   "joint_pipeline_top2_shape": list(result["joint"]["image"]["top2_score"].shape),
                   "joint_pipeline_full_ranking_shape": list(result["joint"]["indicator"]["ranking"].shape)})
    assert report["clinical_to_image_shape"] == [1, 8, 26, 256]
    assert report["image_to_clinical_shape"] == [1, 8, 256, 26]
    assert report["evidence_logits_shape"] == [1, 4]
    assert report["primary_finite"] and report["evidence_finite"]
    assert report["joint_pipeline_finite"] and report["joint_pipeline_weights_unchanged"]
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
