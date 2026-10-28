# Local validation record

- Formal notebook execution: 5/5 code cells completed, 0 error outputs.
- Standalone replay validator: `python scripts/validate_flow_am_dimension_coupling.py --run runs/02-flow-am-dim-formal-8q5obfng --output runs/02-flow-am-dim-formal-8q5obfng/validation.json` — passed, 201 checks and 27 fixed-checkpoint per-path exact replays. Details: `validation.json`.
- Full unit suite: `.venv/bin/python -m unittest discover -s tests -v` — 28 tests passed, 0 failures/errors (including 9 new dimension/coupling test methods).
- Historical 2D four-mode Flow archive: SHA-256 manifest matched original and archived source, executed notebook, and config during the validator and unit tests. Historical result directories remain untouched.

These are local tests in the recorded CPU environment. They are not a claim that GitHub-hosted CI has passed or that numerical checkpoints will be bitwise identical on every platform.
