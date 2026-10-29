# Preserved failed table-reference extension

This is a compact extract of the original v1 run, which stopped at seed 59001,
reference Actor N=50 because the sampled reference-gradient consistency check
failed. `seed-59001/audit-reference-N50/` preserves the points, independent MC
observations, gradient comparisons and failed decision. The protocol, partial
curves and original report are retained without reclassifying the run as a pass.

The complete runtime archive, including the stopped checkpoint, remains on the
training machine. Subsequent quadrature diagnosis did not remove the mismatch.
The separately versioned MC-v2 protocol restarts Actor training at N=0; do not
concatenate these partial curves with the MC-v2 curves.
