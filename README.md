# DR-Bridge Models

Source-derived bidirectional cross-attention, a joint anomaly monitor for image evidence and clinical indicator associations, and evidence-guided attention refinement.

[Chinese guide](README_zh.md)

## Included modules

| File | Function |
| --- | --- |
| `src/drbridge/model.py` | Clinical value, identity and missingness embeddings; patient conditioning by FiLM and gating; two independent cross-attention directions; fusion and DR grading |
| `src/drbridge/image_monitor.py` | One four-class ConvNeXt-Tiny evidence head; anomaly probability; Top-2 image scoring and image-type flags |
| `src/drbridge/indicator_monitor.py` | Coverage-weighted clinical association scores, stable full rankings, Top-6 reference verification |
| `src/drbridge/adjustment.py` | Suppression and enhancement, 8x8 regional coefficients, mapping to 16x16 network positions, spatial renormalization |
| `src/drbridge/pipeline.py` | Joint monitoring, specialist-review flags, frozen-model reinference and paired correction outcomes |
| `src/drbridge/integrity.py` | Checkpoint checksums, exact model-state fingerprints and strict tensor checkpoint loading |

The author's `infer_from_tokens` and `_attention_mask_or_none` methods are preserved. Their AST fingerprints and the source-file fingerprints are recorded in `PROVENANCE.json`. The constructor and input wrappers expose explicit configuration instead of distributing a private training recipe. Input geometry follows the manuscript's 26 clinical variables and 16x16 image positions.

## Release scope

This is an executable **model implementation**, not the complete trained experimental system. It includes the published mechanisms, tensor interfaces, tests and runnable examples. Patient records, annotations, trained checkpoints, fitted preprocessing statistics, private training schedules and unpublished deployment settings are not included. Configuration values in `config/example.json` are software-example values, not the study's training configuration.

The examples and validation reports use non-patient inputs and untrained networks. They verify software behavior, not the manuscript's clinical accuracy, QWK or workload outcomes. Reported study metrics require the corresponding trained artifacts and independent evaluation data.

## Installation

Python 3.10 or later is required. Install a PyTorch build appropriate for the available hardware first. The CPU validation environment used PyTorch 2.7.1 and torchvision 0.22.1:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[backbones]"
```

For CUDA, install the matching CUDA-enabled PyTorch/torchvision build using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install this package. The code uses tensor devices and does not contain hard-coded GPU IDs. No pretrained weights are downloaded automatically.

## Run and verify

From the repository root:

```bash
python scripts/build_manifest.py --check
python scripts/verify_release.py
python -m unittest discover -s tests -v
python examples/run_demo.py --device cpu
python scripts/smoke_backbones.py --device cpu
```

With a CUDA-enabled environment, `--device cuda` can be used for the demo and backbone check. CUDA execution is supported by the interfaces; the included validation report identifies the hardware actually tested.

The lightweight demo does not require timm. It uses a small test encoder and supplied four-class probability vectors to exercise routing and reinference. `smoke_backbones.py` instantiates actual ConvNeXt-Small and ConvNeXt-Tiny models without pretrained weights, verifies the published tensor geometry, and runs their complete joint-monitor/refinement pipeline.

## Joint monitor rules

1. **Image branch.** Class order is valid evidence, imaging artifact or noise, individual fundus characteristic, and other indeterminate factor. The continuous anomaly probability is the sum of the three anomalous-class probabilities. The image score is the mean of the largest two available anomaly probabilities; a single available item uses its own probability. An empty evidence set has no score.
2. **Indicator branch.** Average clinical-to-image attention across heads, weight each spatial location by the fraction of its grid-cell area covered by the evidence region, and sum spatially for each clinical query. Rank all 26 scores using input-index order for ties. The reference set is Scr, eGFR, UA, HbA1c and uAlb. No reference hit within the Top-6 marks an association anomaly; any hit passes the regional reference-coverage check. Aggregate location flags within the paired patient record.
3. **Automatic handling.** Imaging artifacts/noise and individual fundus characteristics supply suppression locations; valid evidence supplies enhancement locations. Other indeterminate image factors retain the initial prediction and enter specialist review. Indicator association anomalies remain available for specialist review alongside any applicable image attention refinement.
4. **Spatial refinement.** For each coarse region, take the maximum anomaly-probability times coverage and the maximum validity-probability times coverage in their respective sets. Combine them as `r = (1 - lambda * a) * (1 + eta * e)`. Both effects contribute when their spatial extents overlap. Map each coefficient to its corresponding 2x2 fine-grid block, multiply the original attention, and renormalize over image keys. The implementation adds `log(r)` to attention logits and checks equivalence to post-softmax reweighting.

The image score, type flags, clinical association state and disease grade are separate outputs. The package retains all evidence types rather than introducing an unpublished exclusive image-category reduction rule. An independently specified image-level category rule and matching physician reference labels are needed when evaluating a single four-class image label.

## Input contract

- Primary model: normalized image tensor `[B, 3, 512, 512]`, processed clinical tensor `[B, 26]`, and optional missingness mask `[B, 26]`.
- Image monitor: selected, padded and normalized evidence images `[M, 3, 224, 224]`.
- Joint monitor: four-class probabilities `[B, E, 4]`, present mask `[B, E]`, and evidence boxes `[B, E, 4]` in **model-input xyxy coordinates**.
- Spatial grid: 256 row-major fine positions and 64 spatially adjacent coarse regions. Consecutive flattened groups of four are not used.
- Clinical names: the exact 26-field order used by the checkpoint. The example schema is not a substitute for checkpoint metadata.
- Optional actual laboratory values are passed separately from standardized model inputs. The code never relabels standardized values as raw measurements.

Evidence-region selection and fitted full-image/clinical preprocessing remain upstream inputs to this release. Their private settings are not guessed by the public package.

For an end-to-end call with selected evidence images, use `drbridge.pipeline.infer_with_joint_monitor`. To use already computed four-class outputs, use `frozen_refinement`. Both preserve initial outputs outside the automatic path. They do not modify original pixels, clinical measurements or primary-model weights.

## Integrity

The release payload is integrity-checked via SHA-256 manifest. Run `python scripts/verify_release.py` to confirm no file has been altered. Hashes establish package integrity, not clinical validity.

See [method mapping](METHOD_MAP.md), [validation report](VALIDATION_REPORT.md) and [release disclosure](RELEASE_SCOPE.md).

