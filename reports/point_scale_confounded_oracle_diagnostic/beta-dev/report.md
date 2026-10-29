# Learned-QAM beta calibration (development only)

Development seed: `59000`; Actor/TD updates: `300`.
Only Learned-QAM was trained. Beta was applied directly as `terminal=-beta*g_L`; the fail-closed MC guard recorded zero reference calls.

Selected by frozen N=300 final-bank ordering: success descending, collision ascending, discounted return descending, beta ascending.

| beta | success | collision | discounted return |
|---:|---:|---:|---:|
| 4 | 53.2471% | 46.7529% | 0.000936 |
| 8 | 59.2529% | 40.7471% | 0.101164 |
| 12 | 62.8418% | 37.1582% | 0.163466 |
| 16 | 65.4053% | 34.5947% | 0.208087 |
| 24 | 58.1787% | 41.8213% | 0.105310 |
| 32 | 48.0225% | 51.9775% | -0.043422 |
| 48 | 31.3477% | 68.6523% | -0.289917 |

Frozen beta: **16**.
The development seed has appeared in earlier method development, so this is a calibration choice, not an independent confirmation.

The calibration endpoint N=300 matches the formal comparison endpoint; this differs from the initial pasted draft's N=100 sweep and avoids training-horizon mismatch.
