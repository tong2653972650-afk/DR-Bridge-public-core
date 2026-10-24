"""Coordinate coverage and the published 16x16-to-8x8 spatial mapping."""

import torch


def validate_attention(attention):
    if attention.ndim != 4 or not attention.is_floating_point():
        raise ValueError("attention must be floating point [B, H, 26, 256]")
    if attention.shape[-2:] != (26, 256):
        raise ValueError("Expected 26 clinical queries and 256 image keys")
    if not attention.shape[0] or not attention.shape[1]:
        raise ValueError("Batch and head dimensions must be nonempty")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("Attention must be finite and nonnegative")
    sums = attention.float().sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3):
        raise ValueError("Attention must sum to 1 over spatial keys; use evaluation mode")


def validate_present(present, shape, device):
    if present.shape != shape or present.dtype != torch.bool or present.device != device:
        raise ValueError("present must be a same-device bool mask matching evidence axes")


def box_coverage(boxes, present, *, image_size=(512, 512), grid=(16, 16)):
    """Fraction of each grid cell covered by each input-space xyxy box."""
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B, E, 4]")
    validate_present(present, boxes.shape[:2], boxes.device)
    if len(image_size) != 2 or len(grid) != 2:
        raise ValueError("image_size and grid must have two dimensions")
    if any(type(v) is not int or v <= 0 for v in (*image_size, *grid)):
        raise ValueError("Image and grid dimensions must be positive integers")
    height, width = image_size
    rows, columns = grid
    if not torch.isfinite(boxes[present]).all():
        raise ValueError("Present evidence boxes must be finite")
    work = torch.where(present[..., None], boxes.float(), torch.zeros_like(boxes.float()))
    x1, y1, x2, y2 = work.unbind(-1)
    x1, x2 = x1.clamp(0, width), x2.clamp(0, width)
    y1, y2 = y1.clamp(0, height), y2.clamp(0, height)
    if ((x2 <= x1) | (y2 <= y1))[present].any():
        raise ValueError("Present boxes must have positive area inside the input image")
    yy, xx = torch.meshgrid(torch.arange(rows, device=boxes.device),
                            torch.arange(columns, device=boxes.device), indexing="ij")
    cell_w, cell_h = width / columns, height / rows
    gx, gy = xx.flatten().float()*cell_w, yy.flatten().float()*cell_h
    dx = (torch.minimum(x2[..., None], gx+cell_w)-torch.maximum(x1[..., None], gx)).clamp_min(0)
    dy = (torch.minimum(y2[..., None], gy+cell_h)-torch.maximum(y1[..., None], gy)).clamp_min(0)
    return (dx*dy/(cell_w*cell_h)).clamp(0, 1)*present[..., None]


def pool_to_regions(attention):
    """Sum neighboring 2x2 spatial positions; retain total attention mass."""
    if attention.shape[-1] != 256:
        raise ValueError("The fine spatial axis must contain 256 positions")
    grid = attention.reshape(*attention.shape[:-1], 8, 2, 8, 2)
    return grid.sum(dim=(-3, -1)).reshape(*attention.shape[:-1], 64)


def expand_region_coefficients(coefficients):
    if coefficients.ndim != 2 or coefficients.shape[-1] != 64:
        raise ValueError("coefficients must be [B, 64]")
    grid = coefficients.reshape(-1, 8, 8)
    return grid.repeat_interleave(2, -2).repeat_interleave(2, -1).reshape(-1, 256)

