"""Non-patient inputs used only for software verification."""

import torch
from torch import nn
from drbridge.model import DRBridgeModel, ModelConfig

FEATURE_NAMES = (
    "diabetes_duration", "height", "weight", "systolic_bp", "diastolic_bp",
    "potassium", "sodium", "chloride", "total_co2", "total_calcium", "Scr",
    "urea", "glucose", "plasma_osmolality", "urinary_creatinine", "eGFR",
    "UA", "triglycerides", "total_cholesterol", "HDL-C", "LDL-C", "HbA1c",
    "uAlb", "random_c_peptide", "homocysteine", "adiponectin",
)


class FeatureInfo:
    def channels(self):
        return [8]


class FixtureImageEncoder(nn.Module):
    feature_info = FeatureInfo()

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, image):
        return [self.conv(image)]


class FixtureEvidenceEncoder(nn.Module):
    def forward(self, image):
        return image.mean((-2, -1))


def make_model():
    torch.manual_seed(17)
    config = ModelConfig(backbone="convnext_small", token_dim=32, num_heads=4,
                         clinical_layers=2, fusion_layers=1, dropout=0.0, head_hidden_dim=32)
    return DRBridgeModel(config, image_backbone=FixtureImageEncoder()).eval()


def sample_inputs():
    torch.manual_seed(19)
    image = torch.randn(3, 3, 16, 16)
    clinical = torch.randn(3, 26)
    probabilities = torch.tensor([
        [[0.03, 0.92, 0.03, 0.02], [0.94, 0.02, 0.02, 0.02]],
        [[0.02, 0.02, 0.02, 0.94], [0.94, 0.02, 0.02, 0.02]],
        [[0.94, 0.02, 0.02, 0.02], [0.94, 0.02, 0.02, 0.02]],
    ])
    boxes = torch.tensor([[[0., 0., 64., 64.], [128., 128., 192., 192.]]]*3)
    present = torch.ones(3, 2, dtype=torch.bool)
    return image, clinical, probabilities, present, boxes

