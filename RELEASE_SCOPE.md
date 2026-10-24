# Release Scope

## Published here

The source-derived attention computation, public image-evidence classifier architecture, Top-2 aggregation, Top-6 association verification, spatial-control equations, frozen reinference, input contracts and verification utilities are executable.

## Supplied by the caller

- Trained primary-model and four-class-monitor checkpoints with SHA-256 digests.
- Model configuration matching those checkpoints.
- Clinical field identities in checkpoint order, fitted imputation and standardization artifacts, and original measurements when needed for review.
- Selected evidence images, corresponding input-coordinate boxes and present masks from the explanatory workflow.
- Attention-control settings and any independently specified single-label image-category aggregation used for image-level classification evaluation.

## Not distributed

Patient data, individual annotations, trained model weights, institutional identifiers, private storage paths, fitted patient-derived statistics, unpublished selection settings and the complete training/search recipe are not part of this release.

The example settings are intentionally separate from the withheld study configuration. The published mechanism is retained; this release does not claim to reproduce trained study performance without the missing trained artifacts. Existing manuscript claims and code availability statements should describe this public-core scope accurately.

The existing MIT license from the author-supplied code is retained. Third-party backbones and dependencies remain subject to their own licenses.

