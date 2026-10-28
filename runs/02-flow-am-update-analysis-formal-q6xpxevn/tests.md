# Fixed-update reanalysis validation

- Executed main Notebook 2: 6/6 code cells completed, 0 error outputs. The execution used `PAGR_DIM_SOURCE_RUN=runs/02-flow-am-dim-formal-8q5obfng` and recorded zero new Actor optimizer steps and zero new checkpoint evaluations.
- `.venv/bin/python -m unittest discover -s tests -v`: **33 tests passed**, 0 failures/errors. Five tests in `tests/test_fixed_update_analysis.py` cover actual N/K/B and Adam step counts; shared initialization; per-seed reward/J/KL signs and decomposition; distinct 512-path curve and 4096-path final streams; saved analysis tables; output provenance; original file and archive hashes.
- `.venv/bin/python scripts/validate_flow_am_dimension_coupling.py --run runs/02-flow-am-dim-formal-8q5obfng --output runs/02-flow-am-update-analysis-formal-q6xpxevn/source-replay-validation.json`: **201 checks passed**, with **27 fixed-checkpoint per-path exact replays**. The validator selected the archived original notebook definitions to validate the unchanged historical training run, rather than misidentifying the revised main notebook source as its trainer.
- The analysis run's `source-manifest.json` records SHA-256 for the original config, completion, environment, audit, learning curves, final table, diagnostics, budgets, and all reused data, frozen models, and actor checkpoints. The original training directory and prior archived results were not edited.

These are local validations, not a claim of GitHub-hosted CI passing or bitwise replay on every platform.
