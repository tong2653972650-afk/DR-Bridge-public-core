# Manuscript to Code Mapping

| Manuscript mechanism | Implementation | Verification |
| --- | --- | --- |
| Clinical value, identity and missingness tokens | `ClinicalTokenTransformer` | Shape, missingness and finite-output tests |
| Patient-specific FiLM and channel gating | `DRBridgeModel.encode` | Numerical comparison against the author source |
| Independent clinical-to-image and image-to-clinical attention | `DRBridgeModel.infer_from_tokens` | Author-method fingerprint, directional normalization and gradient tests |
| 26 clinical queries and 256 fine image locations | Primary model and attention validators | Actual ConvNeXt-Small forward check at 512×512 |
| Four evidence categories and one softmax vector | `ImageEvidenceMonitor` | Four-logit forward, probability and loss-decrease tests |
| Continuous anomaly probability | `summarize_image_evidence` | Sum of the three anomalous categories |
| Top-2 image score | `summarize_image_evidence` | Two-item, one-item, padding and empty-input tests |
| Head-averaged, coverage-weighted clinical scores | `verify_indicator_associations` | Analytically checked scores and boundary coverage |
| Stable Top-6 reference verification | `verify_indicator_associations` | Zero-hit, one-hit, tie-order and field-order tests |
| Clinical reference set | `REFERENCE_INDICATORS` | Scr, eGFR, UA, HbA1c and uAlb |
| Independent image and association flags | `run_joint_monitor` | Parallel automatic/referral path tests |
| Common interference suppression and valid-evidence enhancement | `build_attention_control` | Exact regional coefficient and overlap tests |
| 8×8 coarse control to 16×16 network attention | Spatial mapping functions | Mass preservation and nonconsecutive spatial-index tests |
| Frozen-model reinference | `frozen_refinement` | Unchanged weight/buffer hashes; changed attention and outputs |
| Initial predictions retained outside automatic handling | `_refine_encoded` | Bitwise equality checks |
| Complete correction, partial improvement and other outcomes | `correction_outcomes` | Disjoint paired-outcome partition |

The package does not encode study outcome percentages as targets or manufacture an accuracy improvement. Algorithm tests exercise the mechanism on controlled inputs. The clinical interpretation of association anomalies remains a specialist task.

