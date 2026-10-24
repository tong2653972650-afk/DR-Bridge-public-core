"""Evidence-guided spatial control with frozen-model reinference hooks."""

from dataclasses import dataclass
import math
import torch

from .image_monitor import EvidenceClass, summarize_image_evidence
from .spatial import box_coverage, expand_region_coefficients, validate_attention


@dataclass(frozen=True)
class AdjustmentConfig:
    suppression_strength: float
    enhancement_strength: float

    def __post_init__(self):
        if not math.isfinite(self.suppression_strength) or not 0 <= self.suppression_strength < 1:
            raise ValueError("suppression_strength must be finite and in [0, 1)")
        if not math.isfinite(self.enhancement_strength) or self.enhancement_strength < 0:
            raise ValueError("enhancement_strength must be finite and nonnegative")


def build_attention_control(probabilities, present, boxes, config, *, image_size=(512, 512)):
    summary = summarize_image_evidence(probabilities, present)
    if boxes.device != probabilities.device or boxes.shape[:2] != present.shape:
        raise ValueError("Boxes, probabilities and masks must share evidence axes and device")
    coverage = box_coverage(boxes, present, image_size=image_size, grid=(8, 8))
    labels = summary["class_labels"]
    enabled = summary["automatic_eligible"]
    suppress = ((labels == EvidenceClass.ARTIFACT_OR_NOISE) |
                (labels == EvidenceClass.INDIVIDUAL_FUNDUS)) & present & enabled[:, None]
    enhance = (labels == EvidenceClass.VALID) & present & enabled[:, None]
    q, v = summary["anomaly_probability"], summary["valid_probability"]
    if probabilities.shape[1]:
        a = (q[..., None]*coverage*suppress[..., None]).amax(1)
        e = (v[..., None]*coverage*enhance[..., None]).amax(1)
    else:
        a = probabilities.new_zeros((probabilities.shape[0], 64))
        e = a.clone()
    regional = (1-config.suppression_strength*a)*(1+config.enhancement_strength*e)
    fine = expand_region_coefficients(regional)
    if not torch.isfinite(fine).all() or (fine <= 0).any():
        raise ValueError("Spatial coefficients must be finite and positive")
    return {"regional_suppression": a, "regional_enhancement": e,
            "regional_coefficients": regional, "fine_coefficients": fine,
            "region_bias": fine.log(), "regional_coverage": coverage,
            "automatic_eligible": enabled}


def reweight_attention(attention, fine_coefficients):
    """Reference post-softmax form, equivalent to adding log(r) to logits."""
    validate_attention(attention)
    if fine_coefficients.shape != (attention.shape[0], 256):
        raise ValueError("fine_coefficients must be [B, 256]")
    if fine_coefficients.device != attention.device:
        raise ValueError("Attention and coefficients must share a device")
    if not torch.isfinite(fine_coefficients).all() or (fine_coefficients <= 0).any():
        raise ValueError("Coefficients must be finite and positive")
    weighted = attention.float()*fine_coefficients.float()[:, None, None, :]
    refined = (weighted/weighted.sum(-1, keepdim=True)).to(attention.dtype)
    neutral = fine_coefficients.eq(1).all(-1)
    return torch.where(neutral[:, None, None, None], attention, refined)

