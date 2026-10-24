from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Explicit settings; no private training recipe is supplied as a default."""

    backbone: str
    token_dim: int
    num_heads: int
    clinical_layers: int
    fusion_layers: int
    dropout: float
    head_hidden_dim: int
    clinical_dim: int = 26
    num_classes: int = 5
    image_grid_size: int = 16

    def __post_init__(self) -> None:
        for name in ("token_dim", "num_heads", "clinical_layers", "fusion_layers", "head_hidden_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (self.clinical_dim, self.num_classes, self.image_grid_size) != (26, 5, 16):
            raise ValueError("The released interface uses 26 clinical inputs, 5 grades and a 16x16 grid")


class ClinicalTokenTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(1, config.token_dim)
        self.feature_embedding = nn.Embedding(config.clinical_dim, config.token_dim)
        self.missing_embedding = nn.Embedding(2, config.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim, nhead=config.num_heads,
            dim_feedforward=config.token_dim * 4, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.clinical_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, values, missing_mask=None):
        if values.ndim != 2 or not values.is_floating_point():
            raise ValueError("values must be floating point [B, 26]")
        batch_size, feature_count = values.shape
        if feature_count != self.feature_embedding.num_embeddings:
            raise ValueError("Expected 26 clinical variables")
        if missing_mask is None:
            missing_mask = ~torch.isfinite(values)
        if missing_mask.shape != values.shape or missing_mask.dtype != torch.bool:
            raise ValueError("missing_mask must be bool [B, 26]")
        missing_mask = missing_mask.to(values.device)
        if (~torch.isfinite(values) & ~missing_mask).any():
            raise ValueError("Nonfinite values must be marked missing")
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        feature_ids = torch.arange(feature_count, device=values.device)
        feature_ids = feature_ids.unsqueeze(0).expand(batch_size, feature_count)
        tokens = (
            self.value_embedding(values.unsqueeze(-1))
            + self.feature_embedding(feature_ids)
            + self.missing_embedding(missing_mask.long())
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return encoded[:, 0], encoded[:, 1:]


class DRBridgeModel(nn.Module):
    """Author cross-attention computation with explicit public interfaces."""

    def __init__(self, config: ModelConfig, *, image_backbone=None) -> None:
        super().__init__()
        self.config = config
        c = config
        if image_backbone is None:
            import timm
            image_backbone = timm.create_model(
                c.backbone, pretrained=False, features_only=True, out_indices=(-1,)
            )
        self.image_backbone = image_backbone
        image_channels = image_backbone.feature_info.channels()[-1]
        self.image_channels = image_channels
        self.image_proj = nn.Sequential(
            nn.Conv2d(image_channels, c.token_dim, kernel_size=1),
            nn.BatchNorm2d(c.token_dim), nn.GELU(),
        )
        self.clinical_encoder = ClinicalTokenTransformer(c)
        self.film_gamma = nn.Linear(c.token_dim, c.token_dim)
        self.film_beta = nn.Linear(c.token_dim, c.token_dim)
        self.gate = nn.Sequential(nn.Linear(c.token_dim, c.token_dim), nn.Sigmoid())
        self.clinical_to_image_attn = nn.MultiheadAttention(
            c.token_dim, c.num_heads, dropout=c.dropout, batch_first=True
        )
        self.image_to_clinical_attn = nn.MultiheadAttention(
            c.token_dim, c.num_heads, dropout=c.dropout, batch_first=True
        )
        self.norm_img = nn.LayerNorm(c.token_dim)
        self.norm_cli = nn.LayerNorm(c.token_dim)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=c.token_dim, nhead=c.num_heads,
            dim_feedforward=c.token_dim * 4, dropout=c.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=c.fusion_layers)
        self.classifier = nn.Sequential(
            nn.Linear(c.token_dim * 3, c.head_hidden_dim), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.head_hidden_dim, c.num_classes),
        )

    @staticmethod
    def _attention_mask_or_none(mask: torch.Tensor) -> torch.Tensor | None:
        return None if torch.count_nonzero(mask).item() == 0 else mask

    def encode(self, image, clinical, clinical_missing_mask=None):
        image_feat = self.image_backbone(image)[-1]
        if image_feat.ndim == 4 and image_feat.shape[1] != self.image_channels:
            image_feat = image_feat.permute(0, 3, 1, 2).contiguous()
        image_feat = self.image_proj(image_feat)
        image_feat = F.adaptive_avg_pool2d(
            image_feat, (self.config.image_grid_size, self.config.image_grid_size)
        )
        batch_size, channels, height, width = image_feat.shape
        clinical_cls, clinical_tokens = self.clinical_encoder(clinical, clinical_missing_mask)
        gamma = self.film_gamma(clinical_cls).view(batch_size, channels, 1, 1)
        beta = self.film_beta(clinical_cls).view(batch_size, channels, 1, 1)
        image_feat = image_feat * (1 + gamma) + beta
        image_tokens = image_feat.flatten(2).transpose(1, 2)
        image_tokens = image_tokens * self.gate(clinical_cls).unsqueeze(1)
        return {"image_tokens": image_tokens, "clinical_tokens": clinical_tokens, "image_hw": (height, width)}

    def infer_from_tokens(
        self,
        image_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
        region_bias: torch.Tensor | None = None,
        biomarker_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, image_count, _ = image_tokens.shape
        clinical_count = clinical_tokens.shape[1]
        if region_bias is None:
            region_bias = image_tokens.new_zeros((batch, image_count))
        if biomarker_bias is None:
            biomarker_bias = clinical_tokens.new_zeros((batch, clinical_count))
        c2i_mask = region_bias.unsqueeze(1).expand(batch, clinical_count, image_count)
        i2c_mask = biomarker_bias.unsqueeze(1).expand(batch, image_count, clinical_count)
        c2i_mask = c2i_mask.repeat_interleave(self.config.num_heads, dim=0)
        i2c_mask = i2c_mask.repeat_interleave(self.config.num_heads, dim=0)
        clinical_out, c2i = self.clinical_to_image_attn(
            query=clinical_tokens,
            key=image_tokens,
            value=image_tokens,
            attn_mask=self._attention_mask_or_none(c2i_mask),
            need_weights=True,
            average_attn_weights=False,
        )
        clinical_guided = self.norm_cli(clinical_out + clinical_tokens)
        image_out, i2c = self.image_to_clinical_attn(
            query=image_tokens,
            key=clinical_tokens,
            value=clinical_tokens,
            attn_mask=self._attention_mask_or_none(i2c_mask),
            need_weights=True,
            average_attn_weights=False,
        )
        image_guided = self.norm_img(image_out + image_tokens)
        fusion_tokens = self.fusion_encoder(torch.cat([clinical_guided, image_guided], dim=1))
        fused = torch.cat(
            [clinical_guided.mean(1), image_guided.mean(1), fusion_tokens.mean(1)], dim=1
        )
        return {"logits": self.classifier(fused), "attention_c2i": c2i, "attention_i2c": i2c}

    def forward(self, image, clinical, return_attention=False, *, clinical_missing_mask=None):
        encoded = self.encode(image, clinical, clinical_missing_mask)
        result = self.infer_from_tokens(encoded["image_tokens"], encoded["clinical_tokens"])
        if return_attention:
            result["image_hw"] = encoded["image_hw"]
            return result["logits"], result
        return result["logits"]
