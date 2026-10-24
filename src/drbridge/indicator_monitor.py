"""Coverage-weighted clinical association ranking and Top-6 verification."""

import torch

from .spatial import validate_attention, validate_present

REFERENCE_INDICATORS = ("Scr", "eGFR", "UA", "HbA1c", "uAlb")


def verify_indicator_associations(attention_c2i, fine_coverage, present, feature_names):
    validate_attention(attention_c2i)
    batch, _, clinical_count, spatial_count = attention_c2i.shape
    if fine_coverage.ndim != 3 or fine_coverage.shape[0] != batch or fine_coverage.shape[-1] != spatial_count:
        raise ValueError("fine_coverage must be [B, E, 256]")
    validate_present(present, fine_coverage.shape[:2], attention_c2i.device)
    if fine_coverage.device != attention_c2i.device:
        raise ValueError("Coverage and attention must share a device")
    names = tuple(feature_names)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Feature names must be nonempty strings")
    if len(names) != clinical_count or len(set(names)) != clinical_count:
        raise ValueError("feature_names must uniquely identify the 26 inputs in model order")
    if not all(name in names for name in REFERENCE_INDICATORS):
        raise ValueError("feature_names must contain Scr, eGFR, UA, HbA1c and uAlb")
    if not torch.isfinite(fine_coverage[present]).all():
        raise ValueError("Present coverage values must be finite")
    if ((fine_coverage[present] < 0) | (fine_coverage[present] > 1)).any():
        raise ValueError("Coverage must be in [0, 1]")
    if (fine_coverage[present].sum(-1) <= 0).any():
        raise ValueError("Each verified location must cover at least one spatial cell")
    coverage = torch.where(present[..., None], fine_coverage, torch.zeros_like(fine_coverage))
    scores = torch.einsum("bkn,ben->bek", attention_c2i.float().mean(1), coverage.float())
    ranking = scores.argsort(dim=-1, descending=True, stable=True)
    top6 = ranking[..., :6]
    reference = torch.tensor([names.index(x) for x in REFERENCE_INDICATORS], device=scores.device)
    hits = (top6[..., None] == reference).any(-1).sum(-1)
    zero_hit = (hits == 0) & present
    return {"scores": scores, "ranking": ranking, "top6_indices": top6,
            "reference_hits": hits.masked_fill(~present, -1),
            "association_anomaly_by_region": zero_hit,
            "association_anomaly": zero_hit.any(-1),
            "has_verified_location": present.any(-1), "feature_names": names}
