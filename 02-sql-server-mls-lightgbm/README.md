# 02 · In-Database SQL Server Machine Learning Services (`LightGBM` + Dynamic Feature Store)

## Overview & Why I Built This
In enterprise banking environments, exporting millions of customer rows out of SQL Server into external Python servers via flat files every month was a major operational bottleneck.

At **dataEvo** (2015–2020), I architected an end-to-end **in-database ML pipeline** using **SQL Server Machine Learning Services (`sp_execute_external_script`)** and dynamic T-SQL metaprogramming across three stored procedures:
1. **[`prc_abt_aggregations.sql`](./prc_abt_aggregations.sql) — Metadata-Driven Dynamic SQL Feature Store:**
   - Instead of maintaining thousands of lines of repetitive `SUM()`/`AVG()` SQL, you declare a `1/0` configuration matrix specifying which summary statistics (`avg`, `sum`, `min`, `max`, `stdev`, `range`) to extract per column and whether the column comes from the raw table (`abt`) or the normalized table (`abt_norm`).
   - Automatically generates a companion `nonz` (non-zero count) feature for normalized columns so sparsity/activity signals aren't lost when values `<= 1` are normalized.
2. **[`prc_mdl_model.sql`](./prc_mdl_model.sql) — Zero-Export In-Database LightGBM Training:**
   - Computes stratified positive/negative sampling ratios inside T-SQL, streams the training dataframe directly into embedded Python via `sp_execute_external_script`, runs 5-fold stratified LightGBM CV with early stopping, and serializes the trained booster (`model_to_string()`) + out-of-fold Gini telemetry directly into `dbo.mdl_modelos`.
3. **[`prc_mdl_score.sql`](./prc_mdl_score.sql) — Memory-Bounded Multi-Model Batch Scoring:**
   - Accepts a JSON payload of serialized LightGBM models, reconstructs each `lgb.Booster` in memory, dynamically calculates the forward target horizon (`dep_fec_target`), and scores the customer base in `@batch_size` chunks so multi-model inference stays strictly within SQL Server RAM limits.

---

## 2026 Modernization Notes
Refactored in 2026 with AI to standardize all T-SQL and embedded Python documentation in English, bind parameters cleanly via `sp_executesql` / `sp_execute_external_script`, and upgrade embedded Python to **Pandas 2.x / LightGBM 4.x**.
