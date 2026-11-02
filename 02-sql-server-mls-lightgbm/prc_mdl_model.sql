/*
===============================================================================
prc_mdl_model.sql — In-Database LightGBM CV & Model Training via SQL Server MLS
Originally engineered at dataEvo (2015-2020) for zero-export in-database ML training.
Reviewed & refactored in 2026.

Overview:
Trains a binary LightGBM propensity model directly inside SQL Server Machine Learning
Services (`sp_execute_external_script`), eliminating network/CSV export bottlenecks:
  1. Computes stratified positive/negative sampling ratios from `dbo.tgt_targets`.
  2. Runs 5-fold stratified LightGBM CV with early stopping to find optimal rounds & Gini.
  3. Fits the final Booster, counts active split features, and serializes the model
     (`model_to_string()`) + telemetry directly into `dbo.mdl_modelos`.
===============================================================================
*/

CREATE OR ALTER PROCEDURE dbo.prc_mdl_modelo_flg_devo1 (
    @target         NVARCHAR(32),
    @fec_target     DATE,
    @meses_eventos  INT,
    @rows_t_max     INT = 99999,
    @rows_f_t_rat   INT = 3,
    @lgb_leaves     INT,
    @lgb_min_data   INT,
    @lgb_bag_frac   REAL,
    @lgb_feat_frac  REAL,
    @lgb_rate       REAL,
    @comentarios    NVARCHAR(MAX)
)
AS
BEGIN
    SET NOCOUNT ON;

    -- Compute lookback window start date for target events
    DECLARE @fec_target_ini DATE = DATEADD(MONTH, -@meses_eventos + 1, @fec_target);

    DECLARE @rows_t_all INT;
    DECLARE @rows_f_all INT;
    DECLARE @rows_t_sam INT;
    DECLARE @rows_f_sam INT;
    DECLARE @rows_t_rat REAL;
    DECLARE @rows_f_rat REAL;

    -- Count total positive (1) and negative (0) target instances in the event window
    SELECT @rows_t_all = COUNT(*)
    FROM dbo.tgt_targets
    WHERE fec_target BETWEEN EOMONTH(@fec_target_ini) AND EOMONTH(@fec_target)
      AND tgt_target = @target
      AND tgt_label = 1;

    SELECT @rows_f_all = COUNT(*)
    FROM dbo.tgt_targets
    WHERE fec_target BETWEEN EOMONTH(@fec_target_ini) AND EOMONTH(@fec_target)
      AND tgt_target = @target
      AND tgt_label = 0;

    -- Cap positive rows at @rows_t_max and scale negative sample by @rows_f_t_rat
    SET @rows_t_sam = CASE WHEN @rows_t_max < @rows_t_all THEN @rows_t_max ELSE @rows_t_all END;
    SET @rows_f_sam = CAST(ROUND(@rows_t_sam * @rows_f_t_rat, 0) AS INT);
    SET @rows_t_rat = CAST(@rows_t_sam AS REAL) / NULLIF(CAST(@rows_t_all AS REAL), 0);
    SET @rows_f_rat = CAST(@rows_f_sam AS REAL) / NULLIF(CAST(@rows_f_all AS REAL), 0);

    -- Input query placeholder (populated dynamically in production to join arbitrary feature tables)
    DECLARE @input_data_1 NVARCHAR(MAX) = N'
        -- Production deployment dynamically injects the feature-table JOIN query here,
        -- applying @rows_t_rat and @rows_f_rat for stratified in-database sampling.
    ';

    -- Embedded Python training script executed inside SQL Server MLS
    DECLARE @script NVARCHAR(MAX) = N'
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

target_name = str(df["tgt_target"].iloc[0])
target_col = "tgt_label"
df[target_col] = df[target_col].astype(int)

excluded_cols = {
    "id",
    "fec_datos",
    "fec_target",
    "tgt_label",
    "tgt_target",
    "tgt_json",
}
variables = sorted(col for col in df.columns if col not in excluded_cols)
categoricas_names = [col for col in variables if df[col].dtype.name == "category"]

lgb_train = lgb.Dataset(
    df[variables].to_numpy(),
    label=df[target_col].to_numpy(),
    categorical_feature=categoricas_names or "auto",
    feature_name=variables,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "auc",
    "num_threads": 0,
    "num_leaves": int(lgb_leaves),
    "min_data_in_leaf": int(lgb_min_data),
    "bagging_fraction": float(lgb_bag_frac),
    "bagging_freq": 1,
    "feature_fraction": float(lgb_feat_frac),
    "learning_rate": float(lgb_rate),
    "seed": 7,
    "verbosity": -1,
}

folds = 5
lgb_cv = lgb.cv(
    params,
    lgb_train,
    nfold=folds,
    num_boost_round=5000,
    categorical_feature=categoricas_names or "auto",
    eval_train_metric=True,
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    seed=7,
)

cv_df = pd.DataFrame.from_dict(lgb_cv).sort_values("valid auc-mean", ascending=False).head(1)
mod_cv_tr_gini = float(cv_df["train auc-mean"].iloc[0]) * 2.0 - 1.0
mod_cv_te_gini = float(cv_df["valid auc-mean"].iloc[0]) * 2.0 - 1.0
mod_cv_rounds = int(cv_df.index[0]) + 1

lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=mod_cv_rounds,
    categorical_feature=categoricas_names or "auto",
)

impo_df = pd.DataFrame({
    "feature": lgb_model.feature_name(),
    "split": lgb_model.feature_importance(importance_type="split"),
})
mdl_cols = int((impo_df["split"] > 0).sum())

OutputDataSet = pd.DataFrame([{
    "mdl_target": target_name,
    "mdl_nombre": f"{p_target}_devo1",
    "mdl_comentarios": p_comentarios,
    "mdl_fec_datos": str(df["fec_datos"].max()),
    "mdl_fec_target": str(p_fec_target),
    "mdl_timestamp": datetime.datetime.now(),
    "mdl_t_rows": int(rows_t_sam),
    "mdl_f_rows": int(rows_f_sam),
    "mdl_rounds": mod_cv_rounds,
    "mdl_cols": mdl_cols,
    "mdl_cv_folds": folds,
    "mdl_cv_metrica": "gini",
    "mdl_cv_train": mod_cv_tr_gini,
    "mdl_cv_test": mod_cv_te_gini,
    "mdl_string": lgb_model.model_to_string(),
}])
';

    INSERT INTO dbo.mdl_modelos
    EXECUTE sp_execute_external_script
        @language = N'Python',
        @script = @script,
        @input_data_1 = @input_data_1,
        @input_data_1_name = N'df',
        @params = N'
            @rows_t_sam INT,
            @rows_f_sam INT,
            @lgb_leaves INT,
            @lgb_min_data INT,
            @lgb_bag_frac REAL,
            @lgb_feat_frac REAL,
            @lgb_rate REAL,
            @p_target NVARCHAR(32),
            @p_fec_target DATE,
            @p_comentarios NVARCHAR(MAX)
        ',
        @rows_t_sam = @rows_t_sam,
        @rows_f_sam = @rows_f_sam,
        @lgb_leaves = @lgb_leaves,
        @lgb_min_data = @lgb_min_data,
        @lgb_bag_frac = @lgb_bag_frac,
        @lgb_feat_frac = @lgb_feat_frac,
        @lgb_rate = @lgb_rate,
        @p_target = @target,
        @p_fec_target = @fec_target,
        @p_comentarios = @comentarios;
END;
GO
