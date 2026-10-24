"""Public input adapter for selected local evidence images."""

import numpy as np
import torch
from PIL import Image


def prepare_evidence_image(image):
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(image, ((side-width)//2, (side-height)//2))
    square = square.resize((224, 224), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(square, dtype=np.float32, copy=True)).permute(2, 0, 1)/255
    mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.229, .224, .225])[:, None, None]
    return (tensor-mean)/std

