"""
DEVOPTIMIZER MM7 — Multi-Model LightGBM Hyperparameter Optimizer & Feature Selector
Originally engineered at dataEvo (2015-2020) for retail banking cross-sell / up-sell propensity modeling.
Reviewed & refactored in 2026 (Python 3.12 / Pandas 2.x / NumPy 2.x / LightGBM 4.x).

Key Capabilities:
- Scales across an arbitrary portfolio of binary propensity models (12 banking products by default).
- Rule-based feature exclusion (exact match, prefix match, and substring inclusion/exclusion).
- Fast single-snapshot feature decorrelation (strips zero-variance and collinear features above |r| > threshold).
- Multi-period stratified downsampling across rolling monthly YYMM snapshots.
- Cross-validated hyperparameter grid search with early stopping (LightGBM 4.x callback API).
- Split-gain feature pruning (discards zero-gain features prior to production training to slim downstream ETL).
- Automated month-over-month drift diagnostics (Gini delta, top-10 feature shift, and hyperparameter comparison).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

# Technical Note on LightGBM Column Ordering:
# The feature list (`variables`) passed into `x_train` must keep a deterministic order between
# training and deployment, because NumPy array extraction relies on positional column indexing.
# Since candidate features are selected on the newest monthly snapshot (`periods_list[0]`),
# newer snapshots must be a subset (or exact match) of historical snapshot schemas.

DEFAULT_MODELS: tuple[str, ...] = (
    "FBA",
    "PAQUETE",
    "PFUSD",
    "PFARS",
    "PP",
    "TC",
    "SC",
    "SEG_AP",
    "SEG_AUTO",
    "SEG_DES",
    "SEG_VIDA",
    "SEG_VIVI",
)

DEFAULT_STARTING_PREFIXES: tuple[str, ...] = (
    "POL_",
    "CAM_",
    "SIN_",
    "CIMP_",
    "TB_SCLUB",
    "TB_MOBILE",
    "P8_",
    "TB_PAS_90",
)

DEFAULT_EXACT_MATCHES: tuple[str, ...] = (
    "CUS_NO",
    "PENUMPER",
    "AS_OF_DATE",
    "Record_Count",
    "T_COUNT",
    "F_COUNT",
    "T_COUNT_Sum",
    "F_COUNT_Sum",
    # Features introduced in 1811; excluded until full 12-month history is available:
    "OS_VIDA_FE_ORIGINAL_MTH_Max",
    "OS_VIDA_FE_DESDE_MTH_Max",
    "OS_VIDA_Count",
    "OS_ROBO_Count",
    "OS_AUTO_Count",
    "OS_APER_Count",
    # Deprecated v7.1 onboarding columns:
    "ALTA_PP_MONTO_Sum",
    "ALTA_PP_Sum",
    "ALTA_PP",
    "COMPRA_PAQUETE_TIPO",
    "COMPRA_TC_LIM_DX",
)

DEFAULT_CONTAIN_TOKENS: tuple[str, ...] = (
    "VTC_",
    "_SALDO_",
    "PAS_90",
    "MONTO_MORA",
)

DEFAULT_EXCLUDE_TOKENS: tuple[str, ...] = ("SALDO_TOT_RAT",)


def now_hms() -> str:
    """Return current local time formatted as HH:MM:SS."""
    return str(dt.datetime.now().time().replace(microsecond=0))


def format_duration(seconds_delta: float) -> str:
    """Format elapsed seconds as '00d-00h-00m-00s'."""
    total = int(math.floor(seconds_delta))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days:02d}d-{hours:02d}h-{minutes:02d}m-{seconds:02d}s"


def list_yymm_periods(final_period_yymm: int | str, n_periods: int) -> list[str]:
    """Generate a descending list of `n_periods` monthly YYMM strings starting at `final_period_yymm`."""
    s = f"{int(final_period_yymm):04d}"
    print(f"{now_hms()} - Generating YYMM period window: end={s}, count={n_periods}..")
    current = dt.date(2000 + int(s[:2]), int(s[2:4]), 1)
    periods_list: list[str] = []
    for idx in range(n_periods):
        if idx > 0:
            current = (current - dt.timedelta(days=1)).replace(day=1)
        periods_list.append(f"{current.year % 100:02d}{current.month:02d}")
    print(f"Returned periods_list = {periods_list}")
    return periods_list


def drop_exact_columns(df: pd.DataFrame, matches: Iterable[str]) -> pd.DataFrame:
    """Drop columns matching exact names in `matches`."""
    print("Dropping exact-match exclusion columns..")
    existing = [col for col in matches if col in df.columns]
    missing = [col for col in matches if col not in df.columns]
    if existing:
        df = df.drop(columns=existing)
        for col in existing:
            print(f"  DROPPED: {col}")
    for col in missing:
        print(f"  SKIPPED (not present): {col}")
    return df


def drop_prefix_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    """Drop columns starting with any prefix in `prefixes`."""
    for prefix in prefixes:
        print(f"Dropping columns starting with < {prefix} >..")
        matched = [col for col in df.columns if col.startswith(prefix)]
        df = df.drop(columns=matched)
        print(f"  Matched {len(matched)} columns.")
        preview = matched[:5]
        if preview:
            label = "Showing first 5 dropped:" if len(matched) > 5 else "Dropped columns:"
            print(f"  {label}\n    " + "\n    ".join(preview))
    return df


def drop_contain_except_columns(
    df: pd.DataFrame,
    contain_tokens: Sequence[str],
    exclude_tokens: Sequence[str],
) -> pd.DataFrame:
    """Drop columns matching any token in `contain_tokens`, unless protected by `exclude_tokens`."""
    print(
        f"Filtering columns matching any of {list(contain_tokens)} "
        f"except those matching {list(exclude_tokens)}.."
    )
    matched_contain: set[str] = set()
    for token in contain_tokens:
        matched_contain.update(df.filter(regex=token).columns)

    matched_exclude: set[str] = set()
    for token in exclude_tokens:
        matched_exclude.update(df.filter(regex=token).columns)

    to_drop = sorted(matched_contain - matched_exclude)
    if to_drop:
        print(f"  Removing {len(to_drop)} columns: {to_drop[:10]}")
        df = df.drop(columns=to_drop)
    return df


def decorrelate_features(df: pd.DataFrame, threshold: float = 0.995) -> list[str]:
    """
    Remove constant (zero-variance) features and one of each pair of features whose
    absolute Pearson correlation exceeds `threshold`.
    """
    print(f"Running DECORRELATOR at {now_hms()} (threshold={threshold})..")
    features_df = df.drop(columns=["TARGET"], errors="ignore")
    original_count = features_df.shape[1]

    # Filter out constant columns first
    non_constant_mask = features_df.nunique(dropna=False) > 1
    filtered_df = features_df.loc[:, non_constant_mask]
    constant_removed = original_count - filtered_df.shape[1]

    # Compute absolute correlation matrix and mask the upper triangle (excluding diagonal)
    corr_matrix = filtered_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper_tri.columns if (upper_tri[col] > threshold).any()]
    kept_features = [col for col in filtered_df.columns if col not in to_drop]

    print(f"Finished DECORRELATOR at {now_hms()}:")
    print(f"  Original features : {original_count}")
    print(f"  Removed total     : {original_count - len(kept_features)}")
    print(f"  Removed constant  : {constant_removed}")
    print(f"  Removed collinear : {len(to_drop)} (|r| > {threshold})")
    return kept_features


def sample_by_neg_pos_ratio(
    df: pd.DataFrame,
    ratio_f_t: int,
    seed: int = 7,
) -> pd.DataFrame:
    """Keep all positive rows (TARGET == 1) and sample `ratio_f_t` negatives per positive."""
    df_pos = df[df["TARGET"] == 1].reset_index(drop=True)
    df_neg_pool = df[df["TARGET"] == 0]
    n_neg = min(len(df_neg_pool), len(df_pos) * ratio_f_t)
    df_neg = df_neg_pool.sample(n=n_neg, random_state=seed).reset_index(drop=True)
    combined = pd.concat([df_pos, df_neg], ignore_index=True)

    ratio_pos = len(df_pos) / float(len(combined)) if len(combined) else 0.0
    print(
        f"Sampled by F/T ratio: T={len(df_pos)}, F={len(df_neg)}, "
        f"T_share={ratio_pos:.2%}"
    )
    return combined


def sample_bounded_rows_with_min_ratio(
    df: pd.DataFrame,
    max_rows: int,
    min_ratio_f_t: int,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Cap output at `max_rows` while guaranteeing at least `min_ratio_f_t` negative rows
    for every positive row:
      (T + F = max_rows) and (F = min_ratio_f_t * T) => T = floor(max_rows / (min_ratio_f_t + 1)).
    """
    df_pos = df[df["TARGET"] == 1].reset_index(drop=True)
    df_neg_pool = df[df["TARGET"] == 0]

    if len(df_pos) > 0 and (max_rows - len(df_pos)) / len(df_pos) < min_ratio_f_t:
        target_pos = max(1, math.floor(max_rows / (min_ratio_f_t + 1)))
        df_pos = df_pos.sample(n=min(len(df_pos), target_pos), random_state=seed).reset_index(drop=True)

    target_neg = min(len(df_neg_pool), max(1, max_rows - len(df_pos)))
    df_neg = df_neg_pool.sample(n=target_neg, random_state=seed).reset_index(drop=True)
    combined = pd.concat([df_pos, df_neg], ignore_index=True)

    print(
        f"Bounded sample complete: T={len(df_pos)}, F={len(df_neg)}, "
        f"T/Total={len(df_pos) / len(combined):.2%}, F/T={len(df_neg) / max(1, len(df_pos)):.2f}"
    )
    return combined


def build_lgb_dataset(df: pd.DataFrame, variables: list[str]) -> lgb.Dataset:
    """Impute missing values with -30 sentinel and build a LightGBM Dataset."""
    clean_df = df.fillna(-30)
    y_train = clean_df["TARGET"].to_numpy()
    x_train = clean_df[variables].to_numpy()
    return lgb.Dataset(x_train, label=y_train, feature_name=variables, free_raw_data=False)


def extract_cv_auc_history(gbm_cv: dict[str, list[float]]) -> tuple[list[float], list[float]]:
    """Support both LightGBM <4.0 ('auc-mean') and LightGBM 4.x ('valid auc-mean') CV keys."""
    mean_key = "valid auc-mean" if "valid auc-mean" in gbm_cv else "auc-mean"
    std_key = "valid auc-stdv" if "valid auc-stdv" in gbm_cv else "auc-stdv"
    return gbm_cv[mean_key], gbm_cv[std_key]


def optimize_grid(
    lgb_train: lgb.Dataset,
    model_name: str,
    leaves_vec: Sequence[int],
    mcw_vec: Sequence[int],
    csbt_vec: Sequence[float],
    learning_rate: float,
    xv_folds: int,
) -> pd.DataFrame:
    """Run stratified K-fold cross-validation across the hyperparameter grid."""
    print(f"Optimizing {model_name} at {now_hms()}..")
    grid = list(itertools.product(leaves_vec, mcw_vec, csbt_vec))
    rows: list[dict[str, float | int]] = []

    for idx, (leaves, mcw, csbt) in enumerate(grid, start=1):
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "num_leaves": int(leaves),
            "learning_rate": float(learning_rate),
            "colsample_bytree": float(csbt),
            "min_child_weight": int(mcw),
            "verbosity": -1,
            "seed": 7,
        }
        pct = idx / float(len(grid))
        print(f"  Iteration {idx}/{len(grid)} running at {now_hms()} ({pct:.0%})..")

        gbm_cv = lgb.cv(
            params,
            lgb_train,
            num_boost_round=10000,
            nfold=xv_folds,
            stratified=True,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            seed=7,
        )
        auc_means, auc_stds = extract_cv_auc_history(gbm_cv)
        rows.append(
            {
                "leaves": int(leaves),
                "mcw": int(mcw),
                "csbt": float(csbt),
                "rate": float(learning_rate),
                "rounds": len(auc_means),
                "auc": float(auc_means[-1]),
                "auc_dev": float(auc_stds[-1]),
                "auc-dev": float(auc_means[-1] - auc_stds[-1]),
            }
        )

    results = pd.DataFrame(rows)
    print(f"Optimizer finished at {now_hms()}:")
    print("  Best configuration:")
    print(results.sort_values("auc", ascending=False).head(1).to_string(index=False))
    print(f"  AUC spread (best - worst): {results['auc'].max() - results['auc'].min():.5f}")
    return results


def select_positive_gain_features(
    lgb_train: lgb.Dataset,
    best_pars: pd.DataFrame,
    vars_list: list[str],
) -> list[str]:
    """Train a single model with optimal hyperparameters and retain only features with split gain > 0."""
    print(f"Running VARS_SELECTOR at {now_hms()}..")
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
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=int(best_pars.loc[0, "rounds"]),
        feature_name=vars_list,
    )
    importance = pd.DataFrame(
        {
            "name": model.feature_name(),
            "gain": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)

    vars_used = [str(name).strip() for name in importance.loc[importance["gain"] > 0, "name"]]
    kept_pct = len(vars_used) / float(max(1, len(vars_list)))
    print(
        f"Finished VARS_SELECTOR at {now_hms()}: "
        f"kept {len(vars_used)}/{len(vars_list)} features ({kept_pct:.1%})."
    )
    return vars_used


def check_model_drift(base_dir: Path, model_name: str, periods_list: Sequence[str]) -> None:
    """Compare current period Gini, top-10 features, and hyperparameters against the previous period."""
    if len(periods_list) < 2:
        return
    curr_p, prev_p = periods_list[0], periods_list[1]
    gini_curr_path = base_dir / "GINI" / f"{model_name}_{curr_p}.txt"
    gini_prev_path = base_dir / "GINI" / f"{model_name}_{prev_p}.txt"
    if not (gini_curr_path.exists() and gini_prev_path.exists()):
        return

    print(f"----------- Drift Check: {model_name} ({curr_p} vs {prev_p}) -----------")
    gini_curr = float(pd.read_csv(gini_curr_path)["0"].iloc[0])
    gini_prev = float(pd.read_csv(gini_prev_path)["0"].iloc[0])
    gini_var = 100.0 * (gini_curr - gini_prev) / gini_prev if gini_prev else 0.0
    print(f"Current Gini : {gini_curr * 100:.2f}%")
    print(f"Previous Gini: {gini_prev * 100:.2f}%")
    print(f"Relative Diff: {gini_var:+.2f}%")

    vars_curr = pd.read_csv(base_dir / "VARS" / f"{model_name}_{curr_p}.txt")["0"].head(10).rename("Current")
    vars_prev = pd.read_csv(base_dir / "VARS" / f"{model_name}_{prev_p}.txt")["0"].head(10).rename("Previous")
    print("\nTop 10 Features:")
    print(pd.concat([vars_curr, vars_prev], axis=1).to_string())

    pars_curr = pd.read_csv(base_dir / "PARS" / f"{model_name}_{curr_p}.txt")
    pars_prev = pd.read_csv(base_dir / "PARS" / f"{model_name}_{prev_p}.txt")
    pars_curr["Snapshot"] = "Current"
    pars_prev["Snapshot"] = "Previous"
    print("\nHyperparameters:")
    print(pd.concat([pars_curr, pars_prev], ignore_index=True).to_string(index=False))
    print("----------- End Drift Check -----------")


def run_demo_mode() -> None:
    """Run an end-to-end synthetic demonstration of the optimizer pipeline."""
    print("Running DEVOPTIMIZER MM7 in synthetic --demo mode..")
    rng = np.random.default_rng(7)
    n_rows = 2400
    df = pd.DataFrame(
        {
            "CUS_NO": np.arange(n_rows),
            "POL_LEGACY": rng.normal(size=n_rows),
            "VTC_SPEND": rng.normal(size=n_rows),
            "SALDO_TOT_RAT": rng.normal(size=n_rows),
            "FEAT_1": rng.normal(size=n_rows),
            "FEAT_2": rng.normal(size=n_rows),
            "DEAD_NOISE": np.zeros(n_rows),
        }
    )
    # Create a collinear duplicate and a synthetic target signal
    df["FEAT_1_DUP"] = df["FEAT_1"] * 0.9999 + rng.normal(scale=0.001, size=n_rows)
    logits = 1.2 * df["FEAT_1"] - 0.9 * df["FEAT_2"] + 0.6 * df["SALDO_TOT_RAT"] - 1.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["TARGET"] = (rng.uniform(size=n_rows) < probs).astype(int)

    df = sample_by_neg_pos_ratio(df, ratio_f_t=2)
    df = drop_exact_columns(df, DEFAULT_EXACT_MATCHES)
    df = drop_prefix_columns(df, DEFAULT_STARTING_PREFIXES)
    df = drop_contain_except_columns(df, DEFAULT_CONTAIN_TOKENS, DEFAULT_EXCLUDE_TOKENS)
    vars_decorr = decorrelate_features(df, threshold=0.995)

    lgb_train = build_lgb_dataset(df, vars_decorr)
    results = optimize_grid(
        lgb_train=lgb_train,
        model_name="DEMO_PAQUETE",
        leaves_vec=[16, 32],
        mcw_vec=[10, 20],
        csbt_vec=[0.75, 1.0],
        learning_rate=0.05,
        xv_folds=3,
    )
    best_pars = results.sort_values("auc", ascending=False).head(1).reset_index(drop=True)
    selected_vars = select_positive_gain_features(lgb_train, best_pars, vars_decorr)
    print(f"Selected features for production training: {selected_vars}")
    print("All done! Have a cookie.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DEVOPTIMIZER MM7 — LightGBM Cross-Sell Optimizer")
    parser.add_argument("--demo", action="store_true", help="Run self-contained synthetic demo")
    parser.add_argument("--period", type=int, help="Final period in YYMM format (e.g. 2006)")
    parser.add_argument("--periods", type=int, default=12, help="Number of monthly snapshots to load")
    parser.add_argument("--rows-opti", type=int, default=100_000, help="Max total rows for optimizer dataset")
    parser.add_argument("--base-dir", type=Path, default=Path("Z:/SR.CRM/BASES/MMINER 7/"))
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
        return

    start_time = time.time()
    print("Starting up DEVOPTIMIZER MM7 (dataEvo Cross-Sell Engine)..")
    period = args.period or int(input("Enter model final period (YYMM): "))
    periods = args.periods
    rows_opti = args.rows_opti
    base_dir: Path = args.base_dir

    xv_folds = 5
    opti_ratio_f_t = 2
    leaves_vec = [48, 64, 80]
    mcw_vec = [10, 20, 40]
    csbt_vec = [0.25, 0.5, 0.75]
    learning_rate = 0.02

    periods_list = list_yymm_periods(period, periods)

    for model_name in DEFAULT_MODELS:
        model_start_time = time.time()
        print(f"Starting {model_name.upper()} at {now_hms()}, loading newest snapshot..")

        xinout_prefix = base_dir / "XINOUT" / f"XINOUT_{model_name}_"
        newest_csv = Path(f"{xinout_prefix}{periods_list[0]}.csv")
        df = pd.read_csv(newest_csv, sep=";", decimal=",")

        df = sample_by_neg_pos_ratio(df, opti_ratio_f_t)
        df = drop_exact_columns(df, DEFAULT_EXACT_MATCHES)
        df = drop_prefix_columns(df, DEFAULT_STARTING_PREFIXES)
        df = drop_contain_except_columns(df, DEFAULT_CONTAIN_TOKENS, DEFAULT_EXCLUDE_TOKENS)
        vars_decorr = decorrelate_features(df, threshold=0.995)
        del df

        rows_per_period = math.floor(rows_opti / periods)
        period_frames: list[pd.DataFrame] = []
        for snap in periods_list:
            print(f"Loading snapshot {snap} at {now_hms()}..")
            snap_df = pd.read_csv(
                Path(f"{xinout_prefix}{snap}.csv"),
                sep=";",
                decimal=",",
                usecols=vars_decorr + ["TARGET"],
            )
            snap_df = sample_bounded_rows_with_min_ratio(snap_df, rows_per_period, opti_ratio_f_t)
            period_frames.append(snap_df)

        df_final = pd.concat(period_frames, ignore_index=True)
        lgb_train = build_lgb_dataset(df_final, vars_decorr)
        del df_final

        opti_results = optimize_grid(
            lgb_train, model_name, leaves_vec, mcw_vec, csbt_vec, learning_rate, xv_folds
        )
        opti_results.to_csv(base_dir / "OPTIMIZER_RESULTS" / f"{model_name}_{periods_list[0]}.txt")

        best_pars = opti_results.sort_values("auc", ascending=False).head(1).reset_index(drop=True)
        best_pars.to_csv(base_dir / "PARS" / f"{model_name}_{periods_list[0]}.txt")

        vars_final = select_positive_gain_features(lgb_train, best_pars, vars_decorr)
        pd.DataFrame(vars_final).to_csv(base_dir / "VARS" / f"{model_name}_{periods_list[0]}.txt")

        meta = pd.DataFrame(
            [
                {
                    "N_PERIODS": periods,
                    "ROWS_OPTI": rows_opti,
                    "RATIO_F_T": opti_ratio_f_t,
                    "XV_GINI": float(best_pars.loc[0, "auc"]) * 2.0 - 1.0,
                    "XV_FOLDS": xv_folds,
                    "N_VARS": len(vars_final),
                    "P_TREES": int(best_pars.loc[0, "rounds"]),
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
        meta.to_csv(base_dir / "METAMODEL" / f"{model_name}_{periods_list[0]}_opti_meta.txt", index=False)

    for model_name in DEFAULT_MODELS:
        check_model_drift(base_dir, model_name, periods_list)

    print("All done! Have a cookie.")
    print(f"Total runtime: {format_duration(time.time() - start_time)}")


if __name__ == "__main__":
    main()
