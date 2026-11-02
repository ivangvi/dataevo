# dataEvo & Applied ML Portfolio (2015–2022 · AI-Refactored 2026)

Selected production machine learning, in-database SQL ML, and programmatic ETL engineering work from my 5-year run co-founding **dataEvo** (`2015–2020`) and during grad school at **UCLA Anderson** (`2022`), refactored with AI in 2026 to run cleanly on modern **Python 3.12 / Pandas 2.x / NumPy 2.x / LightGBM 4.x** with unified English documentation.

---

## Repository Contents

| Folder | Original Era | Core Problem Solved & Why It Mattered |
|---|---|---|
| **[`01-lightgbm-cross-sell-engine/`](./01-lightgbm-cross-sell-engine/)** | 2015–2020 (`dataEvo`) | **Scalable 12-Product Banking Propensity Factory (`DEVOPTIMIZER MM7`).** Automated two-stage feature pruning (collinearity filter + split-gain selector), multi-period stratified downsampling, analytical population-prior sample re-weighting (`T_COUNT`/`F_COUNT`), and month-over-month Gini/feature drift monitoring across 12 retail banking products. |
| **[`02-sql-server-mls-lightgbm/`](./02-sql-server-mls-lightgbm/)** | 2015–2020 (`dataEvo`) | **Zero-Export In-Database ML & Dynamic SQL Feature Store.** Declarative `1/0` metadata matrix that dynamically compiles rolling-window SQL aggregations (including a sparsity-preserving `nonz` counter for normalized columns), paired with in-database LightGBM training and memory-bounded batch scoring via SQL Server Machine Learning Services (`sp_execute_external_script`). |
| **[`03-spss-modeler-etl-automation/`](./03-spss-modeler-etl-automation/)** | 2015–2020 (`dataEvo`) | **Programmatic Orchestration for Enterprise SPSS Modeler Streams.** Turned visual drag-and-drop banking streams into batch-executable pipelines with intermediate disk-cache checkpointing (`breaks`), automatic `_3M`/`_6M`/`_9M`/`_12M`/`_DX`/`_AVGDX` field suffixing, 2-month lag routing, and runtime class-balance sampling (`F_SAMPLER`). |
| **[`04-housing-price-shap/`](./04-housing-price-shap/)** | 2022 (`UCLA Anderson`) | **Hedonic Pricing with Variance-Weighted CV & SHAP Explainability.** Physical-constraint anomaly cleaning, time-invariant capture-relative age engineering, chronological 80/20 out-of-time evaluation (`2.15%` R² decay, `12.77%` MAPE), a custom variance-penalized CV metric (`perfo_vw = 2 * R²_valid - R²_train`), and `SHAP TreeExplainer` force/beeswarm attributions. |

---

## What Changed in the 2026 AI Refactor
Rather than leaving 2015–2022 scripts frozen on deprecated library versions or mixed Spanish/English comments, every script in this repository was refactored with AI in 2026:
1. **Stack Modernization (Python 3.12 / Pandas 2.x / NumPy 2.x / LightGBM 4.x):**
   - Upgraded deprecated `DataFrame.append()` calls to `pd.concat()`, `np.bool` to `bool`, `pkg_resources` to `importlib.metadata.version`, parameterized `@params` bindings in SQL Server MLS stored procedures, and updated LightGBM CV early stopping to the modern `callbacks=[lgb.early_stopping(...)]` API.
2. **Self-Contained `--demo` Execution & Clean English Documentation:**
   - Translated all Spanish comments, logs, and variable notes to English, standardized UTF-8 encoding across all SQL scripts, and added `--demo` CLI modes so the Python pipelines run out of the box on synthetic data:

```bash
python 01-lightgbm-cross-sell-engine/lightgbm_optimize.py --demo
python 01-lightgbm-cross-sell-engine/lightgbm_train.py --demo
python 04-housing-price-shap/housing_price_pipeline.py --demo
```
