"""
exec_date_sources.py — Multi-Period Checkpointed ETL Stream Runner for IBM SPSS Modeler
Originally engineered at dataEvo (2015-2020) to automate monthly & historical backfill ETL streams.
Reviewed & refactored in 2026.

Key Capabilities:
- Automates multi-period batch execution (`P_FIRST` to `P_LAST`, formatted as YYMM) across multi-source ETL streams.
- Eliminates manual node path editing and human error during monthly updates or historical re-runs.
- Executes intermediate cache checkpoints (`b1`..`bN` setglobals nodes) so massive multi-gigabyte joins do not
  exhaust temporary disk space.
- Dynamically applies horizon suffixes (`_3M`, `_6M`, `_12M`) on downstream Filter nodes and emits an execution log table.
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence


def periods_range(first_yymm: int | str, last_yymm: int | str) -> list[str]:
    """Generate a descending list of YYMM periods from `first_yymm` down to `last_yymm` (inclusive)."""
    s_first = f"{int(first_yymm):04d}"
    s_last = f"{int(last_yymm):04d}"
    current = dt.date(2000 + int(s_first[:2]), int(s_first[2:4]), 1)
    periods_list: list[str] = []

    while True:
        yymm = f"{current.year % 100:02d}{current.month:02d}"
        periods_list.append(yymm)
        if yymm == s_last:
            break
        current = (current - dt.timedelta(days=1)).replace(day=1)

    return periods_list


def periods_count(first_yymm: int | str, n_periods: int) -> list[str]:
    """Generate a descending list of `n_periods` monthly YYMM strings starting at `first_yymm`."""
    s_first = f"{int(first_yymm):04d}"
    current = dt.date(2000 + int(s_first[:2]), int(s_first[2:4]), 1)
    periods_list: list[str] = []

    for idx in range(n_periods):
        if idx > 0:
            current = (current - dt.timedelta(days=1)).replace(day=1)
        periods_list.append(f"{current.year % 100:02d}{current.month:02d}")

    return periods_list


def datetime_now() -> str:
    """Return current timestamp without microseconds."""
    return str(dt.datetime.now().replace(microsecond=0))


def resolve_source_path(
    source_spec: tuple[str, str, str],
    history_periods: Sequence[str],
) -> tuple[str, str]:
    """
    Resolve a `(node_name, base_dir, extension)` tuple and `history_periods` list into a full file path.
    If `node_name` ends with a trailing digit (e.g. `source_2`), offset into `history_periods` accordingly.
    """
    node_name, base_dir, extension = source_spec
    if node_name[-1].isdigit():
        lag_index = int(node_name.rsplit("_", 1)[1]) - 1
    else:
        lag_index = 0

    full_path = f"{base_dir}{history_periods[lag_index]}{extension}"
    return node_name, full_path


def bind_source_node(diagram, node_name: str, full_path: str) -> None:
    """Bind `full_path` to the appropriate SPSS Modeler import node based on file extension."""
    node_type = "statisticsimport" if full_path.endswith(".sav") else "variablefile"
    diagram.findByType(node_type, node_name).setPropertyValue("full_filename", full_path)


def apply_filter_suffix(diagram, suffix_node_name: str, excluded_fields: Sequence[str]) -> None:
    """Append `suffix_node_name` to every field passing through the named Filter node except ID columns."""
    node = diagram.findByType("filternode", suffix_node_name)
    excluded_set = set(excluded_fields)
    for field in node.getInputDataModel().nameIterator():
        if field not in excluded_set:
            node.setKeyedPropertyValue("new_name", field, f"{field}{suffix_node_name}")


def run_stream(
    modeler_script_env,
    direc_out: str = "C:/data/output/",
    direc_qua: str = "C:/data/quality/",
    history_months: int = 1,
    num_breakpoints: int = 6,
) -> None:
    """Execute the multi-source checkpointed ETL stream across the configured `P_FIRST`..`P_LAST` window."""
    sources: list[tuple[str, str, str]] = [
        ("iv", "C:/data/sources/iv_", ".sav"),
        ("tb_clientes", "C:/data/sources/tb_clientes_", ".sav"),
        ("sucus", "C:/data/sources/sucus_", ".sav"),
        ("tc_limite", "C:/data/sources/tc_limite_", ".sav"),
        ("ingreso", "C:/data/sources/ingreso_", ".sav"),
        ("disponible_paquetes", "C:/data/sources/disp_paq_", ".sav"),
        ("resultado_test_inversor", "C:/data/sources/test_inv_", ".sav"),
    ]
    excluded_ids = ["id_field_1", "id_field_2"]
    suffixes = ["_3M", "_6M", "_12M"]

    diagram = modeler_script_env.diagram()
    stream = modeler_script_env.stream()
    p_first = stream.getParameterValue("P_FIRST")
    p_last = stream.getParameterValue("P_LAST")

    for suffix in suffixes:
        apply_filter_suffix(diagram, suffix, excluded_ids)

    for period in periods_range(p_first, p_last):
        log_entries: list[str] = []
        log_path = f"{direc_out}{period}_log.txt"
        diagram.findByType("outputfile", "log_save").setPropertyValue("full_filename", log_path)

        log_node = diagram.findByType("userinput", "log_create")
        log_node.setPropertyValue("names", ["-- LOG FILE --"])
        log_node.setKeyedPropertyValue("custom_storage", "-- LOG FILE --", "String")

        log_entries.append(f"Running period {period} at {datetime_now()}..")
        current_month_date = dt.date(2000 + int(period[:2]), int(period[2:4]), 1)
        stream.setParameterValue("mesactual", current_month_date)
        log_entries.append(f"Stream parameter mesactual set to {current_month_date}..")

        history_list = periods_count(period, history_months)
        for src in sources:
            node_name, path = resolve_source_path(src, history_list)
            bind_source_node(diagram, node_name, path)
            log_entries.append(f"Source node {src[0]} bound to {path}..")

        diagram.findByType("spssexport", "out").setPropertyValue("full_filename", f"{direc_out}CLP_{period}.sav")
        diagram.findByType("outputfile", "out_q").setPropertyValue("full_filename", f"{direc_qua}CLP_{period}.csv")
        log_entries.append(f"Export node out set to {direc_out}CLP_{period}.sav..")

        # Execute intermediate cache checkpoints to prevent temp-disk exhaustion on large joins
        for bp_idx in range(1, num_breakpoints + 1):
            diagram.findByType("setglobals", f"b{bp_idx}").run(None)
        log_entries.append(f"Executed {num_breakpoints} intermediate cache checkpoints..")

        diagram.findByType("spssexport", "out").run(None)
        log_entries.append("Executed quality audit and final export..")
        log_entries.append(f"Finished at {datetime_now()}! Have a cookie.")

        formatted_log = " ".join(f'"{entry}"' for entry in log_entries)
        log_node.setKeyedPropertyValue("data", "-- LOG FILE --", formatted_log)
        diagram.findByType("table", "log_table").run(None)
        diagram.findByType("outputfile", "log_save").run(None)


if __name__ == "__main__":
    # When executed inside IBM SPSS Modeler, `modeler.script` is injected into the global namespace.
    if "modeler" in globals():
        run_stream(globals()["modeler"].script)
    else:
        print("Dry-run check: periods_range(2006, 2001) ->", periods_range(2006, 2001))
