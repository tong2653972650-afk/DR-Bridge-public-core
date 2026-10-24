"""Build or check the exact-byte manifest after validating LF text files."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_release import payload_files

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".json", ".md", ".yml", ".yaml", ".toml", ".cff", ".txt", ".sha256"}
TEXT_NAMES = {".gitattributes", ".editorconfig", ".gitignore", "LICENSE"}


def validate_text(path, data):
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
        return
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"UTF-8 BOM is not allowed: {path.name}")
    data.decode("utf-8")
    if b"\r" in data:
        raise ValueError(f"Text must use LF line endings: {path.name}")


def manifest_bytes(root):
    paths = sorted(payload_files(root), key=lambda path: path.relative_to(root).as_posix())
    lines = []
    for path in paths:
        data = path.read_bytes()
        validate_text(path, data)
        lines.append(hashlib.sha256(data).hexdigest()+"  "+path.relative_to(root).as_posix()+"\n")
    return "".join(lines).encode("utf-8"), len(paths)


def build_manifest(root=ROOT, *, check=False):
    content, count = manifest_bytes(root)
    path = root/"MANIFEST.sha256"
    if check:
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError("MANIFEST.sha256 is stale; validate changes and rebuild it before committing")
    else:
        path.write_bytes(content)
    return {"files_hashed": count, "text_line_endings": "LF", "manifest": "matched" if check else "written"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(build_manifest(check=args.check), indent=2))
    except (ValueError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
