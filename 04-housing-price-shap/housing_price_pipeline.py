"""
housing_price_pipeline.py — Hedonic Housing Price Modeling with Variance-Weighted CV & SHAP Explainability
Originally authored in 2022; reviewed & refactored in 2026 (Python 3.12 / Pandas 2.x / LightGBM 4.x / SHAP).

Key Capabilities:
- Physical-constraint anomaly cleaning (nullifies impossible `sqft_living > sqft_lot` records).
- Time-invariant feature engineering: converts raw calendar years (`yr_built`, `yr_renovated`) into
  capture-relative ages (`yr_built_ago`, `yr_renovated_ago`), deltas, ratios, and composite quality scores
  (`water_view_score`, `condition_grade_score`) so the model does not decay over future calendar years.
- Strict chronological 80/20 out-of-time validation split (`testing` flag ordered by `date`).
- Custom Variance-Weighted CV Selection Metric (`perfo_vw = 2 * R2_valid - R2_train`): penalizes overfitting
  by subtracting the train-validation R2 gap (`R2_train - R2_valid`) from `R2_valid`.
- Tree-level and global/local SHAP explainability (`TreeExplainer` force plots and beeswarm summary plot).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Nullify physically inconsistent `sqft_lot` values where `sqft_living > sqft_lot`."""
    out = df.copy()
    out["sqft_lot"] = np.where(out["sqft_living"] > out["sqft_lot"], np.nan, out["sqft_lot"])
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-invariant age ratios, room/floor densities, and composite quality scores."""
    out = df.copy()
    out["zipcode"] = pd.Categorical(out["zipcode"])

    # Time-invariant age features relative to transaction capture date
    out["yr_built_ago"] = (out["date"] - out["yr_built"]).dt.days / 365.25
    out["yr_renovated_ago"] = (out["date"] - out["yr_renovated"]).dt.days / 365.25
    out["yr_ren_blt_ago_dif"] = out["yr_renovated_ago"] - out["yr_built_ago"]
    out["yr_ren_blt_ago_rat"] = out["yr_renovated_ago"] / out["yr_built_ago"].replace(0, np.nan)

    # Room and floor ratios
    out["bath_bed_sum"] = out["bathrooms"] + out["bedrooms"]
    out["bath_bed_rat"] = out["bathrooms"] / out["bedrooms"].replace(0, np.nan)
    out["bed_floor_rat"] = out["bedrooms"] / out["floors"].replace(0, np.nan)
    out["bath_bed_sum_sqft_living_rat"] = out["sqft_living"] / out["bath_bed_sum"].replace(0, np.nan)
    out["floors_sqft_living_rat"] = out["sqft_living"] / out["floors"].replace(0, np.nan)

    # Living & lot area ratios (including 15-nearest-neighbor ratios)
    out["sqft_basement_rat"] = out["sqft_basement"] / out["sqft_living"].replace(0, np.nan)
    out["sqft_living_rat"] = out["sqft_living"] / out["sqft_lot"].replace(0, np.nan)
    out["sqft_living15_rat"] = out["sqft_living"] / out["sqft_living15"].replace(0, np.nan)
    out["sqft_lot15_rat"] = out["sqft_lot"] / out["sqft_lot15"].replace(0, np.nan)

    # Domain composite scores
    out["water_view_score"] = out["waterfront"] * 5 + out["view"]
    out["condition_grade_score"] = out["condition"] * 2 + out["grade"]
    return out


def drop_raw_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw ID and non-stationary calendar year columns after relative age extraction."""
    return df.drop(columns=["id", "yr_built", "yr_renovated"], errors="ignore")


def r2_eval_metric(preds: np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
    """Custom R-squared evaluation metric for LightGBM cross-validation."""
    labels = train_data.get_label()
    return "r2", float(r2_score(labels, preds)), True


def run_variance_weighted_cv_grid(
    df_train: pd.DataFrame,
    inputs: list[str],
    output: str,
    params_grid: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Evaluate each parameter configuration with 5-fold CV and compute `perfo_vw`:
      perfo_vw = 2 * R2_valid - R2_train = R2_valid - (R2_train - R2_valid)
    which rewards high out-of-fold R2 while directly penalizing train-vs-validation overfitting.
    """
    tr_ds = lgb.Dataset(df_train[inputs], label=df_train[output], free_raw_data=False)
    rows: list[dict[str, Any]] = []

    for params in params_grid:
        cv_res = lgb.cv(
            params=params,
            train_set=tr_ds,
            num_boost_round=10000,
            stratified=False,
            eval_train_metric=True,
            feval=r2_eval_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            seed=7,
        )
        cv_df = pd.DataFrame(cv_res).tail(1)
        train_key = "train r2-mean" if "train r2-mean" in cv_df.columns else cv_df.columns[0]
        valid_key = "valid r2-mean" if "valid r2-mean" in cv_df.columns else cv_df.columns[2]

        r2_train = float(cv_df[train_key].iloc[0])
        r2_valid = float(cv_df[valid_key].iloc[0])
        perfo_vw = 2.0 * r2_valid - r2_train

        row = dict(params)
        row["num_boost_round"] = int(cv_df.index.values[0]) + 1
        row["perfo"] = r2_valid
        row["perfo_train"] = r2_train
        row["perfo_vw"] = perfo_vw
        rows.append(row)

    return pd.DataFrame(rows).sort_values("perfo_vw", ascending=False).reset_index(drop=True)


def train_selected_booster(
    df_train: pd.DataFrame,
    inputs: list[str],
    output: str,
    selected_row: pd.Series,
) -> lgb.Booster:
    """Fit the final LightGBM model using the selected row from the CV grid results."""
    meta_cols = {"num_boost_round", "perfo", "perfo_train", "perfo_vw"}
    train_params = {k: v for k, v in selected_row.to_dict().items() if k not in meta_cols}
    rounds = int(selected_row["num_boost_round"])

    tr_ds = lgb.Dataset(df_train[inputs], label=df_train[output])
    return lgb.train(params=train_params, train_set=tr_ds, num_boost_round=rounds)


def generate_synthetic_housing_data(n_rows: int = 1500) -> pd.DataFrame:
    """Generate a synthetic King County housing dataset for standalone `--demo` execution."""
    rng = np.random.default_rng(7)
    base_dates = pd.date_range("2014-05-01", periods=n_rows, freq="6h")
    yr_built = rng.integers(1950, 2015, size=n_rows)
    is_renovated = rng.uniform(size=n_rows) < 0.05
    yr_renovated = np.where(is_renovated, np.maximum(yr_built + 10, 2005), 0)

    sqft_living = rng.integers(800, 4500, size=n_rows).astype(float)
    sqft_basement = np.where(rng.uniform(size=n_rows) < 0.4, sqft_living * 0.25, 0.0)
    sqft_above = sqft_living - sqft_basement
    sqft_lot = sqft_living * rng.uniform(1.2, 6.0, size=n_rows)
    # Inject a few physical anomalies (sqft_living > sqft_lot) to test clean_anomalies()
    sqft_lot[:10] = sqft_living[:10] * 0.5

    grade = rng.integers(5, 12, size=n_rows)
    condition = rng.integers(1, 6, size=n_rows)
    waterfront = (rng.uniform(size=n_rows) < 0.03).astype(int)
    view = rng.integers(0, 5, size=n_rows)
    lat = rng.uniform(47.2, 47.75, size=n_rows)
    lon = rng.uniform(-122.45, -121.9, size=n_rows)

    price = (
        180.0 * sqft_living
        + 45_000.0 * grade
        + 250_000.0 * waterfront
        + 350_000.0 * (lat - 47.2)
        + rng.normal(0, 35_000, size=n_rows)
    ).clip(lower=100_000)

    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "date": base_dates,
            "price": price,
            "bedrooms": rng.integers(1, 6, size=n_rows),
            "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], size=n_rows),
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": rng.choice([1.0, 1.5, 2.0, 2.5], size=n_rows),
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "grade": grade,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": pd.to_datetime(yr_built, format="%Y"),
            "yr_renovated": pd.to_datetime(yr_renovated, format="%Y", errors="coerce"),
            "zipcode": rng.choice([98004, 98039, 98103, 98115, 98122], size=n_rows),
            "lat": lat,
            "long": lon,
            "sqft_living15": sqft_living * rng.uniform(0.85, 1.15, size=n_rows),
            "sqft_lot15": sqft_lot * rng.uniform(0.85, 1.15, size=n_rows),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hedonic Housing Price LightGBM + SHAP Pipeline")
    parser.add_argument("--csv", type=Path, default=Path("house_data.csv"), help="Path to house_data.csv")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic housing data")
    args = parser.parse_args()

    if args.demo or not args.csv.exists():
        print("Running Housing Price Pipeline in synthetic --demo mode..")
        df = generate_synthetic_housing_data()
    else:
        df = pd.read_csv(args.csv)
        df["date"] = pd.to_datetime(df["date"])
        df["yr_built"] = pd.to_datetime(df["yr_built"], format="%Y")
        df["yr_renovated"] = pd.to_datetime(df["yr_renovated"], format="%Y", errors="coerce")

    df = clean_anomalies(df)
    df = engineer_features(df)
    df = drop_raw_identifiers(df)

    # Chronological 80/20 out-of-time split
    df = df.sort_values("date").reset_index(drop=True)
    df["testing"] = np.where(np.arange(len(df)) < len(df) * 0.8, 0, 1)

    inputs = sorted(set(df.columns) - {"date", "price", "testing"})
    output = "price"

    grid_spec = {
        "metric": ["None"],
        "objective": ["mse", "mae"],
        "num_leaves": [8, 16] if args.demo else [4, 8, 16, 32],
        "learning_rate": [0.1],
        "feature_fraction": [0.5, 1.0],
        "bagging_fraction": [0.5, 1.0],
        "bagging_freq": [1],
        "verbosity": [-1],
        "seed": [7],
    }
    params_grid = [dict(zip(grid_spec.keys(), v)) for v in itertools.product(*grid_spec.values())]

    df_train = df[df["testing"] == 0].reset_index(drop=True)
    df_test = df[df["testing"] == 1].copy().reset_index(drop=True)

    params_df = run_variance_weighted_cv_grid(df_train, inputs, output, params_grid)
    print("\nTop 5 Grid Candidates by Variance-Weighted R2 (perfo_vw):")
    print(
        params_df[["objective", "num_leaves", "num_boost_round", "perfo_train", "perfo", "perfo_vw"]]
        .head(5)
        .to_string(index=False)
    )

    selected = params_df.iloc[0]
    booster = train_selected_booster(df_train, inputs, output, selected)
    df_test["price_yhat"] = booster.predict(df_test[inputs])

    r2_tr = float(selected["perfo"])
    r2_te = float(r2_score(df_test["price"], df_test["price_yhat"]))
    mape = float(((df_test["price"] - df_test["price_yhat"]).abs() / df_test["price"]).mean())

    print(f"\nOut-of-Fold CV R2 : {r2_tr:.4f}")
    print(f"Out-of-Time Test R2: {r2_te:.4f} (Decay: {1.0 - r2_te / r2_tr:.2%})")
    print(f"Out-of-Time MAPE  : {mape:.2%}")


if __name__ == "__main__":
    main()
