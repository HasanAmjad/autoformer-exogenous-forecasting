# Task 2: Leaderboard Challenge (Autoformer)

One folder per leaderboard attempt. Each folder holds a standalone notebook for exactly what was submitted.

| Folder | Model | P | E | Leaderboard RMSE |
|---|---|---|---|---|
| `v1/` | Autoformer + linear covariate head, seed 0 | 22,224 | 4 | 93.82 |
| `v2/` | Autoformer + non-linear covariate head, 3 seeds refit, floor after prediction | 70,086 | 18 | 71.48 |
| `v3/` | Autoformer + non-linear covariate head + floor layer, 3 seeds refit (**final**) | 70,086 | 18 | **67.22** (rank 3) |
| `analysis/` | baselines, external-data ablation, design comparisons used in the report | | | |

```
Data/          student_train.csv, student_test.csv, optional_external_data.csv
autoformer/    Auto-Correlation and series-decomposition layers from github.com/thuml/Autoformer (MIT)
v1/ v2/ v3/    Task2_vX.ipynb, results/ (trained models, validation tables), figures/
analysis/      analysis.ipynb, results/ (extra runs, tables), figures/
```

**Run:** open a notebook and choose Run All. With `RUN_EXPERIMENTS = False` (the default) it loads the saved models and finishes in about a minute. With `True` it retrains everything from scratch.

Requirements: Python ≥ 3.11, torch, numpy, pandas, matplotlib, jupyter.

The submitted forecast values (`results/submission.txt`) are not committed, in line with the course policy. Each notebook regenerates them in its final section.
