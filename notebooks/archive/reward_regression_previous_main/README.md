# Previous main Notebook 2: single-step noisy reward regression

The previous main [source](../../02_flow_am_dimension_coupling.py) and [fully
executed notebook](../../02_flow_am_dimension_coupling.ipynb) remain at their
original paths **byte-for-byte**. They are archived by reference to avoid a
second large copy and preserve existing links and provenance. The completed
training run `runs/02-flow-am-dim-formal-8q5obfng`, fixed-update read-only
analysis `runs/02-flow-am-update-analysis-formal-q6xpxevn`, their checkpoints,
negative/qualified results and audits have not been overwritten.

This experiment regresses noisy scalar **immediate/terminal reward** in a
one-step bandit where `Q_true=R`. It is an informative terminal-gradient
diagnostic but **not** a TD-trained state-action Q critic and not full QAM.
The new main Notebook 2 is
[02_qam_td_terminal_error.ipynb](../../02_qam_td_terminal_error.ipynb).

Frozen SHA-256 of the previous main files at the start of the TD experiment:

- `02_flow_am_dimension_coupling.py`:
  `8f2d57e4cdc604ae8b145e819f86ef82a346edb468498f231b226f9b4094fb03`
- `02_flow_am_dimension_coupling.ipynb`:
  `4e4a9a2bd65b7ff21f720df495ce46c79d5058ae0e9e452880d0becf4a4dd508`
