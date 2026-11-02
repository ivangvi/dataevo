# 01 · Multi-Product LightGBM Cross-Sell Engine (`DEVOPTIMIZER MM7`)

## Why I Built This (dataEvo, 2015–2020)
At **dataEvo**, retail banking clients needed to maintain and refresh **12 distinct product propensity models** every month (`FBA`, `PAQUETE`, `PFUSD`, `PFARS`, `PP`, `TC`, `SC`, and 5 insurance lines `SEG_*`) across millions of customer-month snapshots. Hand-tuning features and hyperparameters per model every month was slow and error-prone.

I engineered `DEVOPTIMIZER MM7` as an automated, end-to-end model factory with four design choices I was particularly proud of:
1. **Two-Stage Feature Pruning (`decorrelate_features` → `select_positive_gain_features`):**
   - Stage 1 strips constant and collinear features (`|r| > 0.995`) on the newest monthly snapshot before loading multi-period history.
   - Stage 2 trains a single pass with the winning grid hyperparameters and discards every feature with `split gain == 0`, slimming the feature vector before multi-year training and telling the data engineering team which ETL columns could be retired.
2. **Bounded Multi-Period Stratified Sampling (`sample_bounded_rows_with_min_ratio`):**
   - Automatically balances memory ceilings (`100k–120k` rows across 12 monthly snapshots) against class imbalance by guaranteeing a minimum negative-to-positive ratio (`F/T >= 2`) in every monthly slice.
3. **Prior-Probability Sample-Weight Calibration (`compute_population_weights`):**
   - Downsampling majority negatives speeds up training by `10x`, but distorts raw predicted probabilities. `lightgbm_train.py` passes true population counts (`T_COUNT` and `F_COUNT`) through the pipeline and applies analytical observation weights (`w_pos`, `w_neg`) to `lgb.Dataset` so output scores remain calibrated to real population priors.
4. **Automated Month-over-Month Drift Diagnostics (`check_model_drift`):**
   - After training all 12 models, the pipeline automatically compares each model's out-of-fold Gini, top-10 gain features, and hyperparameters against the previous month (`t-1` / `t-2`) to catch silent data drift before deployment.

---

## What Changed in the 2026 AI Refactor
Refactored in 2026 to modernize the stack from **Python 3.7 / Pandas 1.0 / LightGBM 2.x (2020)** to **Python 3.12 / Pandas 2.x / NumPy 2.x / LightGBM 4.x**:
- **Upgraded to modern Python/ML APIs:** Replaced deprecated `DataFrame.append()` with `pd.concat()`, `np.bool` with `bool`, `pkg_resources` with `importlib.metadata.version`, and `early_stopping_rounds` with LightGBM 4.x `callbacks=[lgb.early_stopping(30)]`.
- **Added CLI & self-contained `--demo` mode:** Translated all Spanish comments and logs to English and added `argparse` flags so both scripts can be executed on synthetic banking data in one command without requiring the original network drive:

```bash
python lightgbm_optimize.py --demo
python lightgbm_train.py --demo
```
