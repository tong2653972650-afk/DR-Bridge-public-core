# Validation report

## Release 0.5.2 LF fix

The previous ZIP was committed to a local Git repository with `core.autocrlf=true`, then cloned into LF form. Its exact reported failure was reproduced: `SHA-256 mismatch: PROVENANCE.json`. This release normalizes all text to UTF-8 without BOM and LF, supplies `.gitattributes` and editor settings, and writes manifest/report text with explicit LF endings.

`scripts/build_manifest.py --check` validates the formatting and deterministic exact-byte manifest without rewriting it. Integrity checks continue to detect file changes; no mismatch is ignored.

## Executed checks

- Full CPU unit tests: **49/49 passed**.
- Standard-library integrity tests: **12/12 passed** on each of Python 3.12 and 3.13. Method fingerprints match across both runtimes and both archived author-source copies.
- The eight computational source files retain identical content after LF normalization. Original-source numerical regression maximum absolute difference remains **0.0**.
- Actual ConvNeXt-Small/ConvNeXt-Tiny and the complete joint-monitor/frozen-refinement path returned finite outputs with unchanged primary-model state hashes.
- Real local Git commit/clone checks passed with both `core.autocrlf=true` and `false`; Git index and working-tree text retained LF endings.
- The final payload was committed and cloned again. Manifest/fingerprint checks passed on Python 3.12 and 3.13. Full tests, CPU demo and actual-backbone checks passed from the fresh Git clone.
- The final ZIP was unpacked and verified; an intentional edit triggered SHA-256 mismatch detection.

Python 3.10 and 3.11 are configured in GitHub Actions, not claimed as locally executed. GitHub-hosted CI must still run on the newly uploaded commit; no remote repository was changed by this release build. CUDA execution was not tested.

Inputs are non-patient software fixtures and weights are untrained. These checks validate software behavior, not the manuscript's clinical ACC or QWK. Detailed reports are in `validation/` and the ZIP has an external checksum.
