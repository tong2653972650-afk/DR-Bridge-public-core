"""Public computational core for DR-Bridge."""

from .model import DRBridgeModel, ModelConfig
from .image_monitor import ImageEvidenceMonitor, EvidenceClass, summarize_image_evidence
from .indicator_monitor import verify_indicator_associations
from .adjustment import AdjustmentConfig, build_attention_control
from .pipeline import run_joint_monitor, frozen_refinement

__version__ = "0.5.2"
__all__ = [
    "DRBridgeModel", "ModelConfig", "ImageEvidenceMonitor", "EvidenceClass",
    "summarize_image_evidence", "verify_indicator_associations", "AdjustmentConfig",
    "build_attention_control", "run_joint_monitor", "frozen_refinement",
]

