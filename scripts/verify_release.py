"""Verify every tracked payload file and the preserved author attention method."""

from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_SCHEMA = "python-ast-json-v1"


def payload_files(root):
    generated = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                 ".mypy_cache", ".ruff_cache", ".tox", "build", "dist", "outputs"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in generated or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file() and path.name != "MANIFEST.sha256":
            yield path


def canonical_ast(value):
    """Serialize fields directly instead of relying on version-specific ast.dump."""
    if isinstance(value, ast.AST):
        fields = {}
        for name, child in ast.iter_fields(value):
            # Python 3.12 added this optional field to function/class nodes.
            if name == "type_params" and child == []:
                continue
            fields[name] = canonical_ast(child)
        return {"node": type(value).__name__, "fields": fields}
    if isinstance(value, list):
        return [canonical_ast(child) for child in value]
    if isinstance(value, bytes):
        return {"literal_type": "bytes", "hex": value.hex()}
    if isinstance(value, complex):
        return {"literal_type": "complex", "real": repr(value.real), "imag": repr(value.imag)}
    if value is Ellipsis:
        return {"literal_type": "ellipsis"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported AST field type: {type(value).__name__}")


def ast_digest(node):
    serialized = json.dumps(canonical_ast(node), sort_keys=True, ensure_ascii=True,
                            separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def function_digest(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method_name)
    return ast_digest(method)


def verify(root=ROOT):
    expected = {}
    for line in (root/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        expected[name] = digest
    for name, digest in expected.items():
        path = root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked payload: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in payload_files(root)}
    if actual != set(expected):
        raise ValueError("Payload file list differs from the release manifest")
    provenance = json.loads((root/"PROVENANCE.json").read_text(encoding="utf-8"))
    if provenance.get("fingerprint_schema") != FINGERPRINT_SCHEMA:
        raise ValueError("Unsupported method fingerprint schema")
    for name, digest in provenance["preserved_methods"].items():
        actual_digest = function_digest(root/"src/drbridge/model.py", "DRBridgeModel", name)
        if actual_digest != digest:
            raise ValueError(f"Author method fingerprint mismatch: {name}")
    return {"files_verified": len(expected), "author_attention_fingerprint": "matched"}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
