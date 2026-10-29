# Compact result archive

This is the compact, inspectable result archive for the three-seed 300-update
obstacle QAM experiment. Start with `report.md`, `validation.json`, and
`../../notebooks/02c_qam_point_long_training.ipynb`.

The version-controlled selection includes frozen protocol/source files, all checkpoint
metric tables, gradient-audit points and observations, training logs in CSV,
paired final outcome summaries and regenerated figures. It does not include
the large binary training datasets, optimizer/Actor checkpoints, or per-episode
rollout arrays. Those remain preserved in the complete run directory on the
training machine, where the six final checkpoints were re-evaluated by the
validator. `validation.json` is that completed full-archive check, not a claim
that a compact clone alone can perform the same check without its inputs.

Recreate a complete run with the frozen protocol and the commands in
`../../docs/reproduction.md`. Use a new directory. The initialization-source
setting reuses matching archived pretraining if available; otherwise the same
seed's data and pretraining are generated from scratch.

For compact local viewing, use `analyze_point_long_training.py RUN --plots-only`.
The full analyzer and `validate_point_long_training.py --replay` require the
complete archive. Historical table failures have not been reclassified as
successful validation.
