"""
DEVOPTIMIZER MM7 — Multi-Period Sample-Weighted LightGBM Trainer
Originally engineered at dataEvo (2015-2020) for retail banking cross-sell / up-sell propensity modeling.
Reviewed & refactored in 2026 (Python 3.12 / Pandas 2.x / NumPy 2.x / LightGBM 4.x).

Key Capabilities:
- Ingests optimal hyperparameters (`PARS/`) and pruned feature sets (`VARS/`) produced by `lightgbm_optimize.py`.
- Loads stratified multi-period training windows across 12 monthly snapshots.
- Analytically re-weights downsampled positives and negatives using population `T_COUNT` and `F_COUNT` totals
  so that predicted probabilities remain calibrated to true population priors in production.
- Executes stratified K-fold CV to lock optimal boosting rounds and out-of-fold Gini before fitting the final model.
- Logs full model metadata and runs automated month-over-month Gini / top-feature / parameter drift checks.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from lightgbm_optimize import (
    DEFAULT_MODELS,
    build_lgb_dataset,
    check_model_drift,
    extract_cv_auc_history,
    format_duration,
    list_yymm_periods,
    now_hms,
    sample_bounded_rows_with_min_ratio,
)


def compute_population_weights(
    df: pd.DataFrame,
    total_pop_positives: float,
    total_pop_negatives: float,
) -> np.ndarray:
    """
    Compute per-row observation weights so a downsampled training set reflects true population priors:
      w_pos = Total_Population_Positives / Sampled_Positives
      w_neg = Total_Population_Negatives / Sampled_Negatives
    """
    sampled_pos = max(1, int((df["TARGET"] == 1).sum()))
    sampled_neg = max(1, int((df["TARGET"] == 0).sum()))
    weight_pos = float(total_pop_positives) / float(sampled_pos)
    weight_neg = float(total_pop_negatives) / float(sampled_neg)
    return np.where(df["TARGET"] == 1, weight_pos, weight_neg)


def train_with_cv_gini(
    lgb_train: lgb.Dataset,
    best_pars: pd.DataFrame,
    variables: list[str],
    xv_folds: int,
) -> tuple[lgb.Booster, float, int]:
    """Run stratified CV on the weighted dataset to determine final rounds and Gini, then train final Booster."""
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "learning_rate": float(best_pars.loc[0, "rate"]),
        "num_leaves": int(best_pars.loc[0, "leaves"]),
        "min_child_weight": int(best_pars.loc[0, "mcw"]),
        "colsample_bytree": float(best_pars.loc[0, "csbt"]),
        "verbosity": -1,
        "seed": 7,
    }

    print(f"Starting {xv_folds}-fold cross-validation at {now_hms()}..")
    gbm_cv = lgb.cv(
        params,
        lgb_train,
        num_boost_round=10000,
        nfold=xv_folds,
        stratified=True,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        seed=7,
    )
    auc_means, _ = extract_cv_auc_history(gbm_cv)
    rounds = len(auc_means)
    gini = 2.0 * float(auc_means[-1]) - 1.0

    print(f"Starting final model training ({rounds} rounds, CV Gini={gini:.4f}) at {now_hms()}..")
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=rounds,
        feature_name=variables,
    )
    return booster, gini, rounds


def run_demo_mode() -> None:
    """Run a self-contained demonstration of population-weighted LightGBM training."""
    print("Running DEVOPTIMIZER MM7 Trainer in synthetic --demo mode..")
    rng = np.random.default_rng(42)
    n_rows = 3000
    df = pd.DataFrame(
        {
            "FEAT_1": rng.normal(size=n_rows),
            "FEAT_2": rng.normal(size=n_rows),
            "SALDO_TOT_RAT": rng.normal(size=n_rows),
        }
    )
    logits = 1.3 * df["FEAT_1"] - 1.0 * df["FEAT_2"] + 0.7 * df["SALDO_TOT_RAT"] - 2.0
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["TARGET"] = (rng.uniform(size=n_rows) < probs).astype(int)

    variables = ["FEAT_1", "FEAT_2", "SALDO_TOT_RAT"]
    lgb_train = build_lgb_dataset(df, variables)

    # Simulate true bank population totals (e.g. 15,000 buyers out of 1,500,000 active customers)
    weights = compute_population_weights(df, total_pop_positives=15_000, total_pop_negatives=1_485_000)
    lgb_train.set_weight(weights)

    best_pars = pd.DataFrame([{"rate": 0.05, "leaves": 31, "mcw": 20, "csbt": 0.8}])
    _, gini, rounds = train_with_cv_gini(lgb_train, best_pars, variables, xv_folds=5)
    print(f"Demo weighted model trained: rounds={rounds}, CV Gini={gini:.2%}")
    print("All done! Have a cookie.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DEVOPTIMIZER MM7 — Sample-Weighted LightGBM Trainer")
    parser.add_argument("--demo", action="store_true", help="Run self-contained synthetic demo")
    parser.add_argument("--period", type=int, help="Final period in YYMM format (e.g. 2006)")
    parser.add_argument("--periods", type=int, default=12, help="Number of monthly snapshots to load")
    parser.add_argument("--rows-train", type=int, default=120_000, help="Max total rows for training dataset")
    parser.add_argument("--base-dir", type=Path, default=Path("Z:/SR.CRM/BASES/MMINER 7/"))
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
        return

    start_time = time.time()
    print("Starting up DEVOPTIMIZER MM7 Trainer (dataEvo Cross-Sell Engine)..")
    period = args.period or int(input("Enter model final period (YYMM): "))
    periods = args.periods
    rows_train = args.rows_train
    base_dir: Path = args.base_dir

    xv_folds = 5
    ratio_f_t = 2
    periods_list = list_yymm_periods(period, periods)
    skipped_models: list[str] = []

    for model_name in DEFAULT_MODELS:
        model_start_time = time.time()
        meta_path = base_dir / "METAMODEL" / f"{model_name}_{period}_model_meta.txt"
        if meta_path.is_file():
            print(f"{model_name}: metadata file already exists ({meta_path.name}); skipping.")
            skipped_models.append(model_name)
            continue

        vars_final = pd.read_csv(base_dir / "VARS" / f"{model_name}_{period}.txt")["0"].tolist()
        best_pars = pd.read_csv(base_dir / "PARS" / f"{model_name}_{period}.txt")

        rows_per_period = math.floor(rows_train / periods)
        period_frames: list[pd.DataFrame] = []
        total_pop_t = 0.0
        total_pop_f = 0.0

        for snap in periods_list:
            print(f"Loading training snapshot for {model_name.upper()} ({snap}) at {now_hms()}..")
            csv_path = base_dir / "XINOUT" / f"XINOUT_{model_name}_{snap}.csv"
            snap_df = pd.read_csv(
                csv_path,
                sep=";",
                decimal=",",
                usecols=vars_final + ["TARGET", "T_COUNT", "F_COUNT"],
            )
            snap_df = sample_bounded_rows_with_min_ratio(snap_df, rows_per_period, ratio_f_t)
            total_pop_t += float(snap_df.loc[0, "T_COUNT"])
            total_pop_f += float(snap_df.loc[0, "F_COUNT"])
            period_frames.append(snap_df)

        df_final = pd.concat(period_frames, ignore_index=True)
        lgb_train = build_lgb_dataset(df_final, vars_final)
        weights = compute_population_weights(df_final, total_pop_t, total_pop_f)
        lgb_train.set_weight(weights)
        del df_final

        booster, gini, rounds = train_with_cv_gini(lgb_train, best_pars, vars_final, xv_folds)
        booster.save_model(str(base_dir / "MODELS" / f"{model_name}_{period}.txt"))
        pd.DataFrame(pd.Series([gini])).to_csv(base_dir / "GINI" / f"{model_name}_{period}.txt")

        meta = pd.DataFrame(
            [
                {
                    "N_PERIODS": periods,
                    "ROWS_TRAIN": rows_train,
                    "RATIO_F_T": ratio_f_t,
                    "XV_GINI": gini,
                    "XV_FOLDS": xv_folds,
                    "N_VARS": len(vars_final),
                    "P_TREES": rounds,
                    "P_RATE": float(best_pars.loc[0, "rate"]),
                    "P_LEAVES": int(best_pars.loc[0, "leaves"]),
                    "P_MCW": int(best_pars.loc[0, "mcw"]),
                    "P_COL": float(best_pars.loc[0, "csbt"]),
                    "VER_LGB": pkg_version("lightgbm"),
                    "VER_PD": pd.__version__,
                    "VER_PY": sys.version,
                    "RUNTIME": format_duration(time.time() - model_start_time),
                }
            ]
        )
        meta.to_csv(meta_path, index=False)

    for model_name in DEFAULT_MODELS:
        check_model_drift(base_dir, model_name, periods_list)

    if skipped_models:
        print(f"Models skipped (already trained): {', '.join(skipped_models)}")
    print("All done! Have a cookie.")
    print(f"Total runtime: {format_duration(time.time() - start_time)}")


if __name__ == "__main__":
    main()
