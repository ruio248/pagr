# Historical two-dimensional, four-mode Flow Notebook 2

The source, executed notebook output, and fixed configuration are byte-for-byte
copies of the files at their original paths. The originals remain runnable and
retain all earlier links. SHA-256 values and the source commit are recorded in
`manifest.json`.

The original run directories `runs/02-flow-am-formal-drdkjy4g` and
`runs/02-flow-am-formal-8u0sh2xk` (including the Oracle gain-space and
matched-KL diagnostic outputs) are retained in place, not rerun or modified.
This negative result is a historical experiment with different task settings
and seed numbers; it is not the independent_d2 result in the new notebook.

To rerun the historical notebook from the repository root, execute:

    .venv/bin/python scripts/execute_notebook.py --notebook notebooks/02_flow_am_terminal_gradient.ipynb --output runs/notebooks/flow-am-historical-rerun.ipynb

Choose a fresh output path for each rerun.
