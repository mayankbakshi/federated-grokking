# Release validation

Validation performed on November 18, 2026, on macOS ARM64.

## Figure and saved results

A fresh Python 3.13.1 virtual environment, installed from the figure requirements, reproduced:

- All 160 original source files (60 NPZ diagnostic runs and 100 JSON accuracy-comparison files): SHA-256 hashes matched.
- Complete statistics JSON and per-seed CSV: exact matches.
- Comparison summaries: exact matches. All overlapping model averaging observations agree between the diagnostic battery and the longer accuracy runs across all 20 seeds.
- Three-panel PNG: byte-for-byte match, including every image pixel; 2475 by 1290 pixels.

The PNG is the same figure embedded in the one-page research sample. This verifies reconstruction from the saved results. It does not constitute a new independent run of the full training battery. Exact reproduction establishes consistency with the saved reference, not the absence of analysis errors; the numerical and implementation checks below provide additional, scoped evidence.

The resolved dependency versions are pinned in `requirements-figure-lock.txt`. Run `python -m fedgrok.verify` to repeat these checks. The machine-readable record is `reference/verification.json`.

## Training and kernel implementation

Python 3.11.15, PyTorch 2.12.0, NumPy 2.4.4, and pytest 9.0.3 were used for the training checks, on CPU with one PyTorch thread.

- Eleven standalone tests passed: data partition/split integrity, fixed probe behavior, persistence windows, equality of evaluation grids to the saved grids, exact trace-kernel agreement with full-Jacobian contraction, vectorized/loop gradient agreement, parameter partitioning, degenerate alignment behavior, retained negative/control results, repeatable short training in all regimes, correct aggregation of individual local-only curves, and early individual-client accuracy agreement with saved runs.
- The standalone runner matched the research implementation exactly for central, local, and model averaging runs through recorded step 20, with eight probe examples and two projections. All reported backbone alignment arrays and sampled accuracies matched.
- At the study's full probe size (256) and projection count (8), seed-0 runs in all three regimes exactly matched the first saved kernel checkpoint and all shared accuracy observations through step 20. This is reproducible with `python -m fedgrok.check_training` and is recorded in `reference/training-check.json`.
- The accuracy-only comparison command was executed through step 20 for seed 0. Model averaging and all four independent clients exactly matched the saved global and own-band accuracies at shared observations.

The original library stack for every historical training run was not archived. The checks above validate the released implementation against saved observations at the stated checkpoints. The full 10,000-step diagnostic battery and 100,000-step accuracy study were **not** repeated for this release, and bitwise full-training agreement across platforms is not claimed.

PyTorch reported its standard nested-tensor and CPU-attention vectorization performance warnings. The numerical checks passed.


## Publication review, November 19, 2026

After source-format cleanup, the eleven existing tests passed in Python 3.11.15 with PyTorch 2.12.0, NumPy 2.4.4, and pytest 9.0.3. A further regression check passed for censored accuracy and marker onsets: the lead summary now rejects these inputs instead of assigning zero leads. All 160 saved source-file hashes remain unchanged. Exact figure pixels, PNG bytes, per-seed CSV, comparison summaries, and statistics were reproduced again with the pinned figure environment.

The full training battery was not rerun. Source formatting, clarified comments, the removed unused test import, and the censoring guard do not alter the saved study's numerical results.

## Research summary, November 19, 2026

The one-page study summary now explains the scientific-surrogate motivation and how the pooled-data control narrowed the interpretation. It retains the retrospective lead calculation, shared-probe and calibration limitations, and the distinction between observed generalization and a demonstrated grokking transition. The figure and saved numerical evidence are unchanged.
