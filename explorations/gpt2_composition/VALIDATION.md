# Release checks

Checked on November 19, 2026, using Python 3.11.15, PyTorch 2.12.0,
Transformers 5.14.1, NumPy 2.4.4, Matplotlib 3.10.9, and pytest 9.0.3 on CPU.
These describe the release-check environment, not a recovered historical lockfile.

- All eight bundled evidence files passed their SHA-256 checks. The two original
  training logs and the archived checkpoint-rescore log retain their original bytes.
  Only local filesystem paths in configuration JSON were made relative.
- Regenerating the synthetic world from its recorded configuration reproduced every
  saved array exactly, including the content hash in `meta.json`.
- All five accuracy fields matched between the original and rescored logs at each
  of the 21 common checkpoints.
- The report reproduced the reference numerical summary and CSV. Its regenerated
  PNG matched the reference pixel for pixel in this environment; the figure was
  also inspected for clipping and label legibility.
- Five tests passed: evidence integrity, world regeneration, split separation and
  atomic exposure, answer-only loss semantics, and the full-sequence contrast.
- Both public training commands completed six-step CPU checks with small models,
  generated worlds, checkpoints, and all five evaluation fields. Model downloads
  and online logging were disabled.
- The 23 exported computational definitions match their research-source
  counterparts after removing comments/docstrings and excluding unused analysis
  methods. The standalone runner and output-path helpers are release additions.

The full 150,000-step and 8,000-step experiments were not rerun for this release.
The original large checkpoints are not included, so the released evidence permits
reproduction of the reported analysis, not fresh reevaluation of those weights.
New training uses the archived world and recorded settings with the bounded
evaluator. Numerical trajectories may vary across hardware and library versions.
