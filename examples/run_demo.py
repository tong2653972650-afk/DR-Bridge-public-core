"""Offline, non-patient demonstration of the public computational interfaces."""

from pathlib import Path
import argparse
import json
import sys
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from drbridge.model import DRBridgeModel, ModelConfig
from drbridge.adjustment import AdjustmentConfig
from drbridge.pipeline import frozen_refinement


class FeatureInfo:
    def channels(self):
        return [8]


class ExampleEncoder(nn.Module):
    feature_info = FeatureInfo()

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, images):
        return [self.conv(images)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch environment")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"config/example.json").read_text(encoding="utf-8"))
    names = json.loads((root/"config/clinical_schema.example.json").read_text(encoding="utf-8"))["feature_names"]
    model = DRBridgeModel(ModelConfig(**config["model"]), image_backbone=ExampleEncoder()).to(args.device)
    image = torch.randn(3, 3, 16, 16, device=args.device)
    clinical = torch.randn(3, 26, device=args.device)
    p = torch.tensor([
        [[.03, .92, .03, .02], [.94, .02, .02, .02]],
        [[.02, .02, .02, .94], [.94, .02, .02, .02]],
        [[.94, .02, .02, .02], [.94, .02, .02, .02]],
    ], device=args.device)
    boxes = torch.tensor([[[0., 0., 64., 64.], [128., 128., 192., 192.]]]*3, device=args.device)
    present = torch.ones(3, 2, dtype=torch.bool, device=args.device)
    result = frozen_refinement(model, image, clinical, p, present, boxes, names,
                                AdjustmentConfig(**config["attention_control"]))
    report = {
        "purpose": "software demonstration using non-patient inputs and untrained weights",
        "device": args.device, "torch_version": torch.__version__,
        "top2_scores": result["joint"]["image"]["top2_score"].cpu().tolist(),
        "automatic_eligible": result["joint"]["control"]["automatic_eligible"].cpu().tolist(),
        "specialist_review": result["joint"]["specialist_review"].cpu().tolist(),
        "primary_weights_unchanged": result["state_sha256_before"] == result["state_sha256_after"],
        "max_attention_change": float((result["baseline"]["attention_c2i"]-result["refined"]["attention_c2i"]).abs().max()),
        "max_logit_change": float((result["baseline"]["logits"]-result["refined"]["logits"]).abs().max()),
        "state_sha256": result["state_sha256_after"],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
