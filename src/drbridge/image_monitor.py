"""Four-class image evidence branch and Top-2 image anomaly scores."""

from enum import IntEnum
import torch
from torch import nn

from .spatial import validate_present


class EvidenceClass(IntEnum):
    VALID = 0
    ARTIFACT_OR_NOISE = 1
    INDIVIDUAL_FUNDUS = 2
    OTHER_INDETERMINATE = 3


EVIDENCE_NAMES = (
    "valid_evidence", "imaging_artifact_or_noise",
    "individual_fundus_characteristic", "other_indeterminate_factor",
)


class ImageEvidenceMonitor(nn.Module):
    """Independent ConvNeXt-Tiny features followed by one four-class head."""

    def __init__(self, *, encoder=None, feature_dim=None):
        super().__init__()
        if encoder is None:
            import timm
            encoder = timm.create_model("convnext_tiny", pretrained=False,
                                        num_classes=0, global_pool="avg")
            feature_dim = encoder.num_features
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("feature_dim is required for an injected encoder")
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, 4)

    def forward(self, evidence_images):
        if evidence_images.ndim != 4 or evidence_images.shape[1:] != (3, 224, 224):
            raise ValueError("Expected normalized evidence images [M, 3, 224, 224]")
        features = self.encoder(evidence_images)
        if features.ndim != 2:
            raise ValueError("The evidence encoder must return pooled features [M, D]")
        logits = self.head(features)
        return {"logits": logits, "probabilities": logits.softmax(-1)}


def validate_probabilities(probabilities, present):
    if probabilities.ndim != 3 or probabilities.shape[-1] != 4:
        raise ValueError("probabilities must be [B, E, 4]")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point")
    validate_present(present, probabilities.shape[:2], probabilities.device)
    values = probabilities[present]
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Present probabilities must be finite and in [0, 1]")
    if not torch.allclose(values.sum(-1), torch.ones_like(values.sum(-1)), atol=2e-3, rtol=2e-3):
        raise ValueError("Each present class-probability vector must sum to 1")


def summarize_image_evidence(probabilities, present):
    """Image flags and continuous Top-2 scores from four-class evidence outputs.

    No exclusive image-category reduction rule is imposed: all predicted types
    and their locations remain available. An empty evidence set has no score.
    """
    validate_probabilities(probabilities, present)
    p = torch.where(present[..., None], probabilities, torch.zeros_like(probabilities))
    labels = p.argmax(-1)
    q = p[..., 1:].sum(-1).clamp(0, 1)
    v = p[..., 0]
    count = present.sum(-1)
    scores = p.new_full((p.shape[0],), float("nan"))
    if p.shape[1]:
        largest = q.masked_fill(~present, -torch.inf).topk(min(2, p.shape[1]), dim=-1).values
        largest = largest.masked_fill(~torch.isfinite(largest), 0)
        scores = largest.sum(-1)/count.clamp(max=2).clamp_min(1)
        scores = scores.masked_fill(count == 0, float("nan"))
    flags = torch.stack([((labels == cls) & present).any(-1) for cls in range(4)], -1)
    other = flags[:, EvidenceClass.OTHER_INDETERMINATE]
    manageable = flags[:, EvidenceClass.ARTIFACT_OR_NOISE] | flags[:, EvidenceClass.INDIVIDUAL_FUNDUS]
    return {
        "class_labels": labels.masked_fill(~present, -1),
        "class_flags": flags, "anomaly_probability": q, "valid_probability": v,
        "top2_score": scores, "has_evidence": count > 0,
        "image_anomaly": flags[:, 1:].any(-1),
        "other_indeterminate": other,
        "automatic_eligible": manageable & ~other,
    }


def classify_evidence_batch(monitor, evidence_images, present):
    """Pack only present evidence through the network and scatter outputs back."""
    if evidence_images.ndim != 5 or evidence_images.shape[2:] != (3, 224, 224):
        raise ValueError("evidence_images must be [B, E, 3, 224, 224]")
    validate_present(present, evidence_images.shape[:2], evidence_images.device)
    p = evidence_images.new_zeros((*present.shape, 4))
    if present.any():
        p[present] = monitor(evidence_images[present])["probabilities"]
    return p

