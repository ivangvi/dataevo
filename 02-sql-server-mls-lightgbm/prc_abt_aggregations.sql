/*
===============================================================================
prc_abt_aggregations.sql — Metadata-Driven Dynamic SQL Time-Series Feature Store
Originally engineered at dataEvo (2015-2020) for SQL Server analytical base tables (ABT).
Reviewed & refactored in 2026.

Overview:
Instead of hand-coding hundreds of repetitive SUM/AVG/MIN/MAX/STDEV columns across
multiple lookback windows (3m, 6m, 12m), this stored procedure reads a declarative
1/0 configuration matrix (`#aggregations_config`) that specifies:
  - Which summary statistics to compute per feature (`avg`, `sum`, `min`, `max`, `stdev`, `ran`).
  - Whether to pull the feature from the raw table (`abt`) or the normalized table (`abt_norm`).
  - When `norm = 1`, automatically generates a companion `nonz` (non-zero count) feature
    to preserve sparsity/activity signal that would otherwise be lost during normalization.
===============================================================================
*/

CREATE OR ALTER PROCEDURE dbo.prc_abt_clientes_xm (
    @fec_datos DATE,
    @meses_historia INT
)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @fec_historia DATE = EOMONTH(DATEADD(MONTH, -@meses_historia + 1, @fec_datos));

    -- Guardrail: verify that historical lookback start date exists in the base table
    IF NOT EXISTS (SELECT 1 FROM dbo.abt WHERE fec_datos = @fec_historia)
    BEGIN
        RAISERROR('CRITICAL ERROR: No snapshot found in dbo.abt for @fec_historia (%s). Aborting.', 16, 1);
        RETURN;
    END;

    PRINT 'Snapshot availability check passed for @fec_historia.';

    -- 1. Create the metadata configuration table
    CREATE TABLE #agregaciones (
        variable   NVARCHAR(128) NOT NULL,
        flag_avg   BIT           NOT NULL,
        flag_sum   BIT           NOT NULL,
        flag_min   BIT           NOT NULL,
        flag_max   BIT           NOT NULL,
        flag_stdev BIT           NOT NULL,
        flag_ran   BIT           NOT NULL,
        norm       BIT           NOT NULL,
        expr_avg   NVARCHAR(MAX) NULL,
        expr_sum   NVARCHAR(MAX) NULL,
        expr_min   NVARCHAR(MAX) NULL,
        expr_max   NVARCHAR(MAX) NULL,
        expr_stdev NVARCHAR(MAX) NULL,
        expr_ran   NVARCHAR(MAX) NULL,
        expr_nonz  NVARCHAR(MAX) NULL
    );

    -- 2. Populate declarative feature aggregation matrix (1 = compute, 0 = skip, norm = use abt_norm)
    INSERT INTO #agregaciones (flag_avg, flag_sum, flag_min, flag_max, flag_stdev, flag_ran, variable, norm)
    VALUES
         (1, 0, 0, 0, 0, 0, N'column_01', 0)
        ,(1, 0, 0, 0, 0, 0, N'column_02', 0)
        ,(1, 1, 1, 1, 1, 1, N'column_03', 0)
        ,(1, 1, 1, 1, 0, 1, N'column_04', 1)
        ,(1, 1, 1, 1, 1, 1, N'column_05', 1)
        ,(1, 0, 0, 0, 0, 0, N'column_06', 0)
        ,(1, 1, 0, 0, 0, 0, N'column_07', 1)
        ,(1, 1, 0, 0, 0, 0, N'column_08', 1)
        ,(1, 1, 0, 0, 0, 0, N'column_09', 0)
        ,(1, 0, 0, 0, 0, 0, N'column_10', 0)
        ,(0, 0, 0, 0, 0, 1, N'column_11', 1)
        ,(0, 0, 0, 0, 0, 1, N'column_12', 0)
        ,(1, 1, 1, 1, 1, 1, N'column_13', 0)
        ,(1, 1, 1, 1, 1, 1, N'column_14', 1)
        ,(1, 1, 0, 0, 0, 1, N'column_15', 1)
        ,(1, 0, 0, 0, 0, 1, N'column_16', 0);
        -- Add additional features as needed

    DECLARE @win VARCHAR(12) = CAST(@meses_historia AS VARCHAR(12));

    -- 3. Compile SQL SELECT expressions per feature based on source table (`abt` vs `abt_norm`)
    UPDATE #agregaciones
    SET
        expr_avg = CASE WHEN flag_avg = 1 THEN
            CONCAT('AVG(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') AS ', QUOTENAME(CONCAT(variable, '_', @win, 'm_avg')))
        END,
        expr_sum = CASE WHEN flag_sum = 1 THEN
            CONCAT('SUM(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') AS ', QUOTENAME(CONCAT(variable, '_', @win, 'm_sum')))
        END,
        expr_min = CASE WHEN flag_min = 1 THEN
            CONCAT('MIN(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') AS ', QUOTENAME(CONCAT(variable, '_', @win, 'm_min')))
        END,
        expr_max = CASE WHEN flag_max = 1 THEN
            CONCAT('MAX(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') AS ', QUOTENAME(CONCAT(variable, '_', @win, 'm_max')))
        END,
        expr_stdev = CASE WHEN flag_stdev = 1 THEN
            CONCAT('STDEV(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') AS ', QUOTENAME(CONCAT(variable, '_', @win, 'm_stdev')))
        END,
        expr_ran = CASE WHEN flag_ran = 1 THEN
            CONCAT(
                'MAX(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') - ',
                'MIN(', CASE WHEN norm = 0 THEN 'abt.' ELSE 'abt_norm.' END, QUOTENAME(variable), ') AS ',
                QUOTENAME(CONCAT(variable, '_', @win, 'm_ran'))
            )
        END,
        -- Non-zero observation count preserves activity signal when normalized values <= 1 collapse
        expr_nonz = CASE WHEN norm = 1 THEN
            CONCAT('COUNT(abt_norm.', QUOTENAME(variable), ') AS ', QUOTENAME(CONCAT(variable, '_', @win, 'm_nonz')))
        END;

    -- 4. Assemble dynamic SQL statement across all non-null aggregation expressions
    DECLARE @agg_clauses NVARCHAR(MAX);

    SELECT @agg_clauses = STRING_AGG(expr, N',' + CHAR(13) + CHAR(10) + N'    ')
    FROM (
        SELECT expr_avg   AS expr FROM #agregaciones WHERE expr_avg   IS NOT NULL UNION ALL
        SELECT expr_sum   AS expr FROM #agregaciones WHERE expr_sum   IS NOT NULL UNION ALL
        SELECT expr_min   AS expr FROM #agregaciones WHERE expr_min   IS NOT NULL UNION ALL
        SELECT expr_max   AS expr FROM #agregaciones WHERE expr_max   IS NOT NULL UNION ALL
        SELECT expr_stdev AS expr FROM #agregaciones WHERE expr_stdev IS NOT NULL UNION ALL
        SELECT expr_ran   AS expr FROM #agregaciones WHERE expr_ran   IS NOT NULL UNION ALL
        SELECT expr_nonz  AS expr FROM #agregaciones WHERE expr_nonz  IS NOT NULL
    ) AS active_expressions;

    DECLARE @target_table NVARCHAR(128) = QUOTENAME(CONCAT('abt', @win, 'm'));

    DECLARE @sql NVARCHAR(MAX) = CONCAT(
        N'SELECT', CHAR(13), CHAR(10),
        N'    abt.id_cli_persona,', CHAR(13), CHAR(10),
        N'    MAX(abt.fec_datos) AS fec_datos,', CHAR(13), CHAR(10),
        N'    COUNT(abt.id_cli_persona) AS abt_count_', @win, N'm,', CHAR(13), CHAR(10),
        N'    ', @agg_clauses, CHAR(13), CHAR(10),
        N'INTO #temp_abt', CHAR(13), CHAR(10),
        N'FROM dbo.abt AS abt', CHAR(13), CHAR(10),
        N'INNER JOIN dbo.abt_norm AS abt_norm', CHAR(13), CHAR(10),
        N'    ON abt.id_cli_persona = abt_norm.id_cli_persona', CHAR(13), CHAR(10),
        N'   AND abt.fec_datos = abt_norm.fec_datos', CHAR(13), CHAR(10),
        N'WHERE abt.fec_datos BETWEEN EOMONTH(@p_fec_historia) AND EOMONTH(@p_fec_datos)', CHAR(13), CHAR(10),
        N'  AND abt_norm.fec_datos BETWEEN EOMONTH(@p_fec_historia) AND EOMONTH(@p_fec_datos)', CHAR(13), CHAR(10),
        N'GROUP BY abt.id_cli_persona;', CHAR(13), CHAR(10),
        -- Retain only customers active in the most recent snapshot (@fec_datos)
        N'SELECT * INTO dbo.', @target_table, N' FROM #temp_abt WHERE fec_datos = @p_fec_datos;'
    );

    -- 5. Execute parameterized dynamic SQL
    EXEC sp_executesql
        @stmt = @sql,
        @params = N'@p_fec_historia DATE, @p_fec_datos DATE',
        @p_fec_historia = @fec_historia,
        @p_fec_datos = @fec_datos;
END;
GO
