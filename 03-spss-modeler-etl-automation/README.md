# 03 · Programmatic ETL Stream Orchestration for IBM SPSS Modeler

## Overview & Why I Built This
Many enterprise banks ran their core analytics pipelines inside **IBM SPSS Modeler**—a visual drag-and-drop workflow tool. Every month, analysts had to manually update dozens of file import/export nodes and re-declare derived field suffixes, which was slow and prone to human error or disk-cache crashes during multi-year backfills.

At **dataEvo** (2015–2020), I built a programmatic Python orchestration layer inside SPSS Modeler's scripting API to turn visual streams into deterministic, batch-executable ETL pipelines:

---

### 1. [`exec_date_sources.py`](./exec_date_sources.py) — Multi-Period Checkpointed Stream Runner
Automates monthly production runs and multi-period historical backfills (`P_FIRST` to `P_LAST`) across multi-source joins.
- **Intermediate Cache Checkpointing (`breaks`):** Executes numbered breakpoint nodes (`b1`..`b6`) sequentially to flush intermediate caches to disk so massive multi-source joins never exhaust temporary disk space.
- **Automated Audit Trail:** Generates an in-stream execution log table recording exact timestamps, bound file paths, and stream parameters.

![Multi-Source Checkpointed Stream](https://i.imgur.com/xiP2Fqp.jpg)

---

### 2. [`exec_date_aggs.py`](./exec_date_aggs.py) — Automated 12-Month Rolling Time-Series Aggregations
Executes rolling 3-, 6-, 9-, and 12-month time-series aggregations (`min`, `max`, `sd`, `avg`, `delta`/`_DX`, and `ratio`/`_AVGDX`) across 12 monthly snapshots.
- **Dynamic Filter-Node Field Suffixing:** Iterates over `node.getInputDataModel().nameIterator()` to automatically append `_3M`, `_6M`, `_9M`, `_12M`, `_DX`, and `_AVGDX` suffixes to hundreds of derived features without manual field declarations.

![Rolling 12-Month Time-Series Stream](https://i.imgur.com/Vfu5P4U.jpg)

---

### 3. [`exec_slices.py`](./exec_slices.py) — Multi-Product Lag-Window Router & Dynamic Sampler
Iterates through both a list of monthly snapshots and a list of banking products (`PAQUETE`, `SC`, `PP`, `TC`) to export product-specific training datasets:
- **Automatic Feature-to-Target Lag Routing (`resolve_lagged_source_filename`):** Enforces a strict 2-month observation-to-outcome lag window (`window = 2`) plus source-specific historical offsets (`_T3`, `_T6`, `_T12`) to prevent target leakage.
- **Runtime Class-Balance Calibration (`F_SAMPLER`):** Queries positive buyer counts (`T_COUNT`) from stream globals at runtime and dynamically sets negative sampling counts (`max(2 * T_COUNT, 150,000 - T_COUNT)`) per product.

![Multi-Product Slice Extraction Stream](https://i.imgur.com/18sALSf.jpg)

---

## 2026 Modernization Notes
Refactored in 2026 with AI into modular Python 3 functions with shared `YYMM` period-window helpers, type annotations, English documentation, and standalone CLI dry-run verification.
