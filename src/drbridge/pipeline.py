"""Joint monitoring, parallel review flags and frozen attention refinement."""

from contextlib import contextmanager
import torch

from .adjustment import build_attention_control, reweight_attention
from .image_monitor import summarize_image_evidence, classify_evidence_batch
from .indicator_monitor import verify_indicator_associations
from .integrity import state_dict_sha256
from .spatial import box_coverage


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def run_joint_monitor(attention_c2i, probabilities, present, boxes, feature_names,
                      adjustment_config, *, image_size=(512, 512), raw_clinical_values=None):
    images = summarize_image_evidence(probabilities, present)
    fine_coverage = box_coverage(boxes, present, image_size=image_size, grid=(16, 16))
    indicators = verify_indicator_associations(attention_c2i, fine_coverage, present, feature_names)
    control = build_attention_control(probabilities, present, boxes, adjustment_config,
                                      image_size=image_size)
    if raw_clinical_values is not None:
        if raw_clinical_values.shape != (attention_c2i.shape[0], 26):
            raise ValueError("raw_clinical_values must be [B, 26] in feature_names order")
        raw_clinical_values = raw_clinical_values.detach().clone()
    review = images["other_indeterminate"] | indicators["association_anomaly"]
    return {
        "image": images, "indicator": indicators, "control": control,
        "specialist_review": review,
        "review_image_only": images["other_indeterminate"] & ~indicators["association_anomaly"],
        "review_indicator_only": indicators["association_anomaly"] & ~images["other_indeterminate"],
        "review_both": images["other_indeterminate"] & indicators["association_anomaly"],
        "raw_clinical_values": raw_clinical_values,
        "evidence_boxes": boxes.detach().clone(),
    }


def _refine_encoded(model, encoded, baseline, joint):
    before = state_dict_sha256(model)
    control = joint["control"]
    refined = model.infer_from_tokens(encoded["image_tokens"], encoded["clinical_tokens"],
                                       region_bias=control["region_bias"])
    expected = reweight_attention(baseline["attention_c2i"], control["fine_coefficients"])
    if not torch.allclose(refined["attention_c2i"].float(), expected.float(), atol=3e-5, rtol=3e-4):
        raise RuntimeError("Reinference attention differs from the published reweighting rule")
    enabled = control["automatic_eligible"]
    # Explicitly retain baseline output for cases outside the automatic path.
    for name in ("logits", "attention_c2i", "attention_i2c"):
        mask = enabled.reshape((enabled.shape[0],)+(1,)*(baseline[name].ndim-1))
        refined[name] = torch.where(mask, refined[name], baseline[name])
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Primary model state changed during frozen reinference")
    return {"baseline": baseline, "refined": refined, "joint": joint,
            "baseline_probabilities": baseline["logits"].softmax(-1),
            "refined_probabilities": refined["logits"].softmax(-1),
            "state_sha256_before": before, "state_sha256_after": after}


def frozen_refinement(model, image, clinical, probabilities, present, boxes,
                      feature_names, adjustment_config, *, image_size=(512, 512),
                      clinical_missing_mask=None, raw_clinical_values=None):
    """Use supplied monitor outputs; encode unchanged inputs once and reinfer."""
    with evaluation_mode(model):
        initial_hash = state_dict_sha256(model)
        encoded = model.encode(image, clinical, clinical_missing_mask)
        baseline = model.infer_from_tokens(encoded["image_tokens"], encoded["clinical_tokens"])
        joint = run_joint_monitor(baseline["attention_c2i"], probabilities, present, boxes,
                                   feature_names, adjustment_config, image_size=image_size,
                                   raw_clinical_values=raw_clinical_values)
        result = _refine_encoded(model, encoded, baseline, joint)
        if result["state_sha256_after"] != initial_hash:
            raise RuntimeError("Primary model state changed during encoding")
        return result


def infer_with_joint_monitor(model, image_monitor, image, clinical, evidence_images,
                             present, boxes, feature_names, adjustment_config, **kwargs):
    """Run the four-class network, both monitor branches and frozen refinement."""
    with evaluation_mode(image_monitor):
        probabilities = classify_evidence_batch(image_monitor, evidence_images, present)
    return frozen_refinement(model, image, clinical, probabilities, present, boxes,
                              feature_names, adjustment_config, **kwargs)


def correction_outcomes(initial_grade, refined_grade, reference_grade):
    """Paired image outcomes using absolute distance to the reference grade."""
    if initial_grade.shape != refined_grade.shape or initial_grade.shape != reference_grade.shape:
        raise ValueError("All grade arrays must have the same shape")
    for values in (initial_grade, refined_grade, reference_grade):
        if values.dtype not in (torch.int32, torch.int64) or ((values < 0) | (values > 4)).any():
            raise ValueError("Grades must be integer DR0-DR4 labels")
    e0, e1 = (initial_grade-reference_grade).abs(), (refined_grade-reference_grade).abs()
    complete = (e0 > 0) & (e1 == 0)
    partial = (e1 > 0) & (e1 < e0)
    unchanged = initial_grade == refined_grade
    nonimproving = ~(complete | partial | unchanged)
    return {"complete": complete, "partial": partial, "unchanged": unchanged,
            "nonimproving": nonimproving, "beneficial": complete | partial,
            "correct_to_incorrect": nonimproving & (e0 == 0),
            "incorrect_to_incorrect": nonimproving & (e0 > 0),
            "grade_shift": (refined_grade-initial_grade).abs()}
