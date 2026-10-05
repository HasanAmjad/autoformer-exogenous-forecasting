# Autoformer with Exogenous Covariates for Long-Horizon Forecasting

Coursework for **AI651 – Deep Learning for Space, Time and Graphs** (LUMS, Fall 2026), Assignment 1.

| Folder | Contents |
|---|---|
| `Task 1/` | Design study of decomposition and delay-based mixing (the Autoformer building blocks) against attention and ridge baselines. `Assignment1.ipynb` with the supplied `harness/`; outputs in `results/design/`. |
| `Task 2 - Leaderboard/` | A 168-step Autoformer forecaster with external covariates: `v1/`, `v2/` and `v3/` (one notebook per leaderboard attempt) and `analysis/`. See its README. |
| `report/` | LaTeX source, compiled PDF and figures. |

## Task 2 in brief

* **Final model (`v3`):** an Autoformer (series decomposition + Auto-Correlation; layers from [thuml/Autoformer](https://github.com/thuml/Autoformer), MIT licence) with a non-linear head on the covariates over the forecast horizon and a floor layer. Three seeds are refit on the full history and averaged: 70,086 parameters, 18 epochs.
* **Leaderboard:** 93.82 → 71.48 → **67.22 RMSE (rank 3)** over the three attempts.
* **External data:** covariates help only when they cover the forecast horizon (validation RMSE 56.8 vs 82.5 without them).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas matplotlib jupyter
# open any notebook in "Task 2 - Leaderboard/v1|v2|v3|analysis" and Run All
```

Submitted forecast values are not committed, in line with the course policy. Each notebook regenerates them.

## Reference

Wu, H., Xu, J., Wang, J., & Long, M. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS.
