# Changelog

## 0.5.2 - 2026-10-20

- Fixed exact-byte manifests after Git line-ending normalization. All release text now uses UTF-8 without BOM and LF endings.
- Added repository `.gitattributes` and editor settings to keep checkout text in LF form, including on Windows.
- Added `scripts/build_manifest.py`: validates text formatting, writes deterministic manifests, and checks without silently rewriting them in CI.
- Added LF/BOM/stale-manifest regression tests and Git commit/clone round-trip verification.
- Kept the computational model, monitoring and refinement logic unchanged.

Reference: [Git attributes documentation](https://git-scm.com/docs/gitattributes).

## 0.5.1 - 2026-10-20

- Fixed method-fingerprint verification across Python versions. Python 3.13 changed the default formatting of `ast.dump`, which could reject an unchanged method in a package built with Python 3.12.
- Replaced display-string hashing with the versioned `python-ast-json-v1` schema and regenerated provenance and file manifests.
- Added fingerprint regression tests and a Python 3.10-3.13 standard-library integrity CI matrix.
- Retained the model, monitoring, spatial adjustment and pipeline implementations unchanged.

Reference: [Python AST documentation](https://docs.python.org/3.13/library/ast.html#ast.dump).
