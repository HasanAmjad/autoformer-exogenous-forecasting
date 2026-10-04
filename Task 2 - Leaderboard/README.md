# AI651 PA1 – Task 2 (Leaderboard): Autoformer

* `Task2.ipynb`: the full, executed pipeline (data, periodicity, split, baselines, model, training, ablation, final forecast).
* `task2.py`: the same code as the notebook, as a CLI so seeds can run in parallel:
  `python task2.py baselines`, `python task2.py run --variant pastfuture --seed 0`, `python task2.py final --variant pastfuture --seed 0`.
  `run_all_seeds.sh` runs all 9 (variant, seed) jobs, two at a time.
* `autoformer/`: Auto-Correlation and series-decomposition layers copied from the official implementation,
  https://github.com/thuml/Autoformer (MIT licence). Wu et al., NeurIPS 2021.
* `results/`: per-run metrics, histories and weights (`runs/`), baselines, the ablation LaTeX table, and the submission files.
* `figures/`: numbered PDF figures for the report.

Requirements: Python ≥ 3.11, torch, numpy, pandas, matplotlib (the same environment as Task 1).
Final submission: variant `pastfuture`, seed 0. P and E are printed by the notebook and saved to `results/submission_meta.json`.
