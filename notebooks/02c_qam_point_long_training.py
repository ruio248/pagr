# %% [markdown]
# # Notebook 2 extension: does the QAM gap persist with longer training?
#
# This extends the existing two-dimensional obstacle task to three seeds and
# 300 Actor optimizer updates per method. The environment, offline data recipe,
# frozen Base, deployment sampler, Lean Adjoint, Adam, and TD implementation are
# inherited unchanged from `02_qam_point_obstacle.py`.
#
# Ordinary QAM uses `-beta * grad_a Q_critic` as the terminal adjoint. The
# privileged reference branch uses the current Actor's numerical policy-value
# integral instead. Both branches still perform one TD update per Actor update.
# A GPU accelerates the full value-grid recursion; endpoint gradients retain the
# original NumPy score integral. No new PAGR method is introduced.
#
# The new user-approved protocol checks gradient precision, not scalar Q as a
# veto. Independent MC uses the original NumPy simulator and separate IID
# trajectories. Scalar Q disagreement is retained as a diagnostic. Snapshot
# checks include actual training endpoints; they do not certify every point.
#
# The old N=10 smoke and original audit FAIL are separate historical artifacts.

# %%
import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
if not (ROOT/'configs/point_long_training.json').exists():
    ROOT = ROOT.parent
protocol = json.loads((ROOT/'configs/point_long_training.json').read_text())
display(pd.DataFrame([dict(seeds=protocol['seeds'], actor_updates=protocol['actor_updates'],
                          curve_updates=protocol['curve_updates'],
                          audit_updates=protocol['audit_updates'],
                          final_episodes=protocol['final_episodes'])]))

# %% [markdown]
# Training implementation: `scripts/run_point_long_training.py`.
# Accelerated reference: `scripts/point_reference_accelerated.py`.
# Analysis: `scripts/analyze_point_long_training.py`.
#
# Launch explicitly from the repository root (GPU environment):
#
# ```bash
# .venv/bin/python -u scripts/run_point_long_training.py \
#   --output runs/02-qam-point-obstacle-long-gradient-v1-20261029 --device cuda
# ```
#
# Checkpoints preserve Actor, Critic, target, both Adam states, training RNG,
# and source/protocol hashes. Repeating the command resumes the same protocol;
# a changed scientific source or configuration is rejected.
#
# Comparison uses fixed checkpoints, independent final evaluation, paired test
# noise, and all three seeds. Extra reference compute is reported separately.
# Success thresholds 60/70/80 percent are fixed before observing the curves.

# %%
run = Path(os.environ.get('PAGR_LONG_RUN',
           ROOT/'runs/02-qam-point-obstacle-long-gradient-v1-20261029'))
if (run/'protocol.json').exists():
    display(Markdown((run/'report.md').read_text()) if (run/'report.md').exists()
            else Markdown('Training has started; the aggregate report is pending.'))
else:
    display(Markdown('No local result directory yet. Training can run on Ubuntu; '
                     'set `PAGR_LONG_RUN` to a synced result directory to inspect it.'))

# %%
for filename in ('success-gap-aggregate.csv', 'final-independent.csv',
                 'sample-efficiency.csv', 'gradient-audits.csv'):
    if (run/filename).exists():
        display(Markdown(f'**{filename}**'))
        display(pd.read_csv(run/filename))
for filename in ('learning-curves.png', 'success-gap.png'):
    if (run/filename).exists():
        display(Image(filename=str(run/filename)))

# %% [markdown]
# Interpret early and late gaps separately. An early advantage followed by
# convergence can support an update-efficiency benefit. A persistent advantage
# can support a finite-budget performance benefit. Neither pattern should be
# assumed in advance, and three training seeds remain exploratory evidence.
