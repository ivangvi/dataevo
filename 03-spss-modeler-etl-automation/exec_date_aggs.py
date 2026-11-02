"""
exec_date_aggs.py — Automated 12-Month Rolling Time-Series Aggregation Runner for IBM SPSS Modeler
Originally engineered at dataEvo (2015-2020) for scalable multi-horizon feature generation.
Reviewed & refactored in 2026.

Key Capabilities:
- Automates 12-month rolling window aggregations (`_3M`, `_6M`, `_9M`, `_12M`) across 12 monthly `.sav` snapshots.
- Dynamically renames hundreds of derived summary fields (`min`, `max`, `sd`, `avg`, `delta`/`_DX`, `ratio`/`_AVGDX`)
  on Filter nodes without manual field declarations.
- Loops across `P_FIRST`..`P_LAST` to support both single-month production updates and multi-period historical backfills.
"""

from __future__ import annotations

from exec_date_sources import periods_count, periods_range

ID_FIELD = "IDFIELD"
HORIZON_SUFFIXES: tuple[str, ...] = ("_3M", "_6M", "_9M", "_12M")


def configure_filter_node_renames(diagram, id_field: str = ID_FIELD) -> None:
    """Dynamically append horizon, delta (`_DX`), and average-delta (`_AVGDX`) suffixes on Filter nodes."""
    # 1. Base horizon Filter nodes (_3M, _6M, _9M, _12M)
    for suffix in HORIZON_SUFFIXES:
        node = diagram.findByType("filternode", suffix)
        for field in node.getInputDataModel().nameIterator():
            if field != id_field:
                node.setKeyedPropertyValue("new_name", field, f"{field}{suffix}")

    # 2. Delta (_DX) and Average-Delta (_AVGDX) Filter nodes
    for suffix in HORIZON_SUFFIXES:
        dx_node = diagram.findByType("filternode", f"_DX{suffix}")
        for field in dx_node.getInputDataModel().nameIterator():
            if field != id_field:
                base_name = field.replace("_Sum", "")
                dx_node.setKeyedPropertyValue("new_name", field, f"{base_name}_DX{suffix}")

        avgdx_node = diagram.findByType("filternode", f"_AVGDX{suffix}")
        for field in avgdx_node.getInputDataModel().nameIterator():
            if field != id_field:
                base_name = field.replace("_Mean_Sum", "")
                avgdx_node.setKeyedPropertyValue("new_name", field, f"{base_name}_AVGDX{suffix}")


def run_rolling_aggregations(
    modeler_script_env,
    base_dir: str = "C:/data/snapshots/",
    lookback_months: int = 12,
) -> None:
    """Bind 12 monthly snapshots for each target period in `P_FIRST`..`P_LAST` and run the aggregation stream."""
    diagram = modeler_script_env.diagram()
    stream = modeler_script_env.stream()
    p_first = stream.getParameterValue("P_FIRST")
    p_last = stream.getParameterValue("P_LAST")

    configure_filter_node_renames(diagram)

    for period in periods_range(p_first, p_last):
        history_list = periods_count(period, lookback_months)

        diagram.findByType("statisticsexport", "out").setPropertyValue(
            "full_filename", f"{base_dir}{period}_12M.sav"
        )
        for idx, hist_period in enumerate(history_list, start=1):
            diagram.findByType("statisticsimport", str(idx)).setPropertyValue(
                "full_filename", f"{base_dir}{hist_period}.sav"
            )

        diagram.findByType("dataaudit", "QUALI").run(None)
        diagram.findByType("statisticsexport", "out").run(None)


if __name__ == "__main__":
    if "modeler" in globals():
        run_rolling_aggregations(globals()["modeler"].script)
    else:
        print("Dry-run check: 12-month window from 2006 ->", periods_count(2006, 12))
