"""File, weight and release integrity checks."""

from pathlib import Path
import hashlib
import json
import re
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model):
    """Hash sorted names, tensor metadata and exact CPU bytes, including buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError("Only dense tensor state dictionaries are supported")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        digest.update(metadata.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(model, path, *, expected_sha256):
    """Load an explicitly identified tensor checkpoint, never an arbitrary object."""
    if not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    if file_sha256(path) != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("Expected a tensor state_dict")
    model.load_state_dict(state, strict=True)
    return state_dict_sha256(model)

