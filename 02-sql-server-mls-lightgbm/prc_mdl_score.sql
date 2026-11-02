/*
===============================================================================
prc_mdl_score.sql — Memory-Bounded Multi-Model Batch Scoring in SQL Server MLS
Originally engineered at dataEvo (2015-2020) for scalable in-database LightGBM inference.
Reviewed & refactored in 2026.

Overview:
Scores an arbitrary JSON payload of serialized LightGBM models (`mdl_string`) against
large customer feature tables while strictly bounding RAM usage via `@batch_size`:
  1. Deduplicates `json_modelos` to retain the latest trained model per target.
  2. Reconstructs each `lgb.Booster(model_str=...)` in memory and computes the forward
     prediction horizon (`dep_fec_target`) from the training metadata window.
  3. Loops across `@batch_size` customer chunks and inserts predictions into `dbo.mdl_score`.
===============================================================================
*/

CREATE OR ALTER PROCEDURE dbo.prc_mdl_score (
    @fec_datos    DATE,
    @batch_size   INT,
    @json_modelos NVARCHAR(MAX),
    @comentarios  NVARCHAR(MAX)
)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @script NVARCHAR(MAX) = N'
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import lightgbm as lgb

# Parse JSON model registry and retain only the newest model per target
df_modelos = pd.read_json(json_modelos)
df_modelos = (
    df_modelos.sort_values(["mdl_target", "mdl_timestamp"], ascending=False)
    .drop_duplicates(subset="mdl_target")
    .reset_index(drop=True)
)

dep_fec_datos = datetime.datetime.strptime(str(p_fec_datos), "%Y-%m-%d")
batch_frames = []

for _, mod_row in df_modelos.iterrows():
    fec_datos_train = datetime.datetime.strptime(str(mod_row["mdl_fec_datos"]), "%Y-%m-%d")
    fec_target_train = datetime.datetime.strptime(str(mod_row["mdl_fec_target"]), "%Y-%m-%d")
    horizon_months = (
        (fec_target_train.year - fec_datos_train.year) * 12
        + (fec_target_train.month - fec_datos_train.month)
    )
    dep_fec_target = dep_fec_datos + relativedelta(months=+horizon_months)

    lgb_model = lgb.Booster(model_str=mod_row["mdl_string"])
    variables = lgb_model.feature_name()

    batch_frames.append(
        pd.DataFrame({
            "id": df["id"],
            "fec_datos": dep_fec_datos,
            "fec_target": dep_fec_target,
            "sco_timestamp": datetime.datetime.now(),
            "sco_mdl_nombre": mod_row["mdl_nombre"],
            "sco_mdl_timestamp": mod_row["mdl_timestamp"],
            "sco_mdl_fec_target": mod_row["mdl_fec_target"],
            "sco_target": mod_row["mdl_target"],
            "sco_label": "1",
            "sco_score": lgb_model.predict(df[variables].to_numpy()),
            "sco_comentarios": p_comentarios,
        })
    )

OutputDataSet = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame()
';

    -- Compute total number of customer batches required for @fec_datos
    DECLARE @row_count INT = (
        SELECT COUNT(*)
        FROM dbo.clus_abt
        WHERE per_fec_carga = EOMONTH(@fec_datos)
    );
    DECLARE @batch_count INT = CEILING(CAST(@row_count AS FLOAT) / NULLIF(@batch_size, 0));

    DECLARE @input_data_1 NVARCHAR(MAX);
    DECLARE @batch INT = 1;
    DECLARE @batch_ini INT = 1;
    DECLARE @batch_fin INT;

    WHILE (@batch <= @batch_count)
    BEGIN
        SET @batch_fin = @batch_ini + @batch_size;
        SET @input_data_1 = N'
            -- Production deployment injects the dynamic SQL query selecting rows
            -- BETWEEN @batch_ini AND @batch_fin across the active feature tables.
        ';

        INSERT INTO dbo.mdl_score
        EXECUTE sp_execute_external_script
            @language = N'Python',
            @script = @script,
            @input_data_1 = @input_data_1,
            @input_data_1_name = N'df',
            @params = N'
                @json_modelos NVARCHAR(MAX),
                @p_fec_datos DATE,
                @p_comentarios NVARCHAR(MAX)
            ',
            @json_modelos = @json_modelos,
            @p_fec_datos = @fec_datos,
            @p_comentarios = @comentarios;

        SET @batch_ini = @batch_fin;
        SET @batch = @batch + 1;
    END;
END;
GO
