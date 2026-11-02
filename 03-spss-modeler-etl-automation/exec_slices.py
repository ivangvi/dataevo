"""
exec_slices.py — Multi-Product Lag-Window Router & Dynamic Negative Sampler for IBM SPSS Modeler
Originally engineered at dataEvo (2015-2020) to build product-specific training slices from a master stream.
Reviewed & refactored in 2026.

Key Capabilities:
- Iterates across both a range of monthly periods (`P_FIRST`..`P_LAST`) and a list of banking products (`outs`),
  routing each source node to its appropriate historical lag (`window = 2` months between features and target).
- Applies product-specific eligibility filters (`CONDICION` select node) and derives binary target flags (`COMPRA_<PROD>`).
- Reads positive target counts (`T_COUNT`) from stream globals at runtime and dynamically sets the negative sample size
  (`F_SAMPLER`) to maintain a 2:1 negative-to-positive ratio with a 150,000-row minimum floor.
"""

from __future__ import annotations

from typing import Sequence

from exec_date_sources import periods_count, periods_range

ID_FIELDS = {"PENUMPER", "CUS_NO"}


def apply_lag_renames(diagram, renamers: Sequence[tuple[str, str]]) -> None:
    """Append lag suffixes (`_T3`, `_T6`, `_T12`) to non-ID fields on the specified Filter nodes."""
    for prefix, lag_tag in renamers:
        node = diagram.findByType("filternode", f"{prefix}_{lag_tag}")
        suffix = f"_{lag_tag}"
        for field in node.getInputDataModel().nameIterator():
            if field not in ID_FIELDS:
                node.setKeyedPropertyValue("new_name", field, f"{field}{suffix}")


def resolve_lagged_source_filename(
    source_entry: tuple[int, str, str],
    history_list: Sequence[str],
    feature_target_window: int = 2,
) -> str:
    """
    Compute the historical `.sav` filename for a source node based on its naming conventions:
      - `is_output == 1`: target table (lag = 0); `is_output == 0`: feature table (lag = `feature_target_window`).
      - Trailing digit (e.g. `altas_mm_7_3`): adds `int(digit)` months to the lag offset.
      - Trailing `_norm`: routes to normalized `_norm.sav` file.
      - `CLP_NORM_T12`: explicit 12-month lag override.
    """
    is_output, raw_name, subdir = source_entry
    source_name = raw_name
    extension = ".sav"

    lag_idx = 0 if is_output == 1 else feature_target_window

    if source_name[-1].isdigit():
        lag_idx += int(source_name[-1])
        source_name = source_name[:-2]
    elif source_name.endswith("_norm"):
        source_name = source_name[:-5]
        extension = f"_norm{extension}"

    if source_name.endswith("_"):
        source_name = source_name[:-1]

    if source_name.startswith("CLP_NORM"):
        source_name = "CLP_NORM"
    if raw_name == "CLP_NORM_T12":
        lag_idx = 12

    return f"{subdir}{source_name}_{history_list[lag_idx]}{extension}"


def run_slice_extraction(
    modeler_script_env,
    modeler_api_module,
    direc_inp: str = "C:/data/inputs/",
    direc_out: str = "C:/data/outputs/",
    window: int = 2,
) -> None:
    """Execute the multi-period, multi-product training slice extraction stream."""
    renamers = [("CLP", "T3"), ("CLP", "T6"), ("CLP", "T12")]

    # (is_output: 1=target / 0=feature, node_name, relative_subdir)
    sources: list[tuple[int, str, str]] = [
        (1, "altas_mm_7_3", "MMINER 7/ALTAS/"),
        (1, "altas_mm_7_2", "MMINER 7/ALTAS/"),
        (1, "altas_mm_7_1", "MMINER 7/ALTAS/"),
        (1, "altas_mm_7_", "MMINER 7/ALTAS/"),
        (0, "OSB_OM_12M", "OSB_12M/"),
        (0, "OSA_OM_6M", "OSA_6M/"),
        (0, "OS", "OS/"),
        (0, "QYR_12M", "QYR/"),
        (0, "CLP", "CLP/"),
        (0, "CLP_12M", "CLP/"),
        (0, "CLP_NORM_T3", "CLP/"),
        (0, "CLP_NORM_T6", "CLP/"),
        (0, "CLP_NORM_T12", "CLP/"),
        (0, "VRZ", "VRZ/"),
    ]
    products = ["PAQUETE", "SC", "PP", "TC"]

    diagram = modeler_script_env.diagram()
    stream = modeler_script_env.stream()
    p_first = stream.getParameterValue("P_FIRST")
    p_last = stream.getParameterValue("P_LAST")

    apply_lag_renames(diagram, renamers)

    periods_list = periods_range(p_first, p_last)
    total_lookback = 12 + window

    for idx, period in enumerate(periods_list, start=1):
        stream.setParameterValue("mesactual", f"20{period[:2]}-{period[2:4]}-01")
        print(f"RUNNING PERIOD {period} -- ({idx}/{len(periods_list)})")

        history_list = periods_count(period, total_lookback)

        for src in sources:
            rel_path = resolve_lagged_source_filename(src, history_list, feature_target_window=window)
            full_path = f"{direc_inp}{rel_path}"
            print(f"Binding node {src[1]} -> {full_path}")
            diagram.findByType("statisticsimport", src[1]).setPropertyValue("full_filename", full_path)

        altas_out = f"{direc_inp}MMINER 7/ALTAS/ALTAS_7_{period}_POST_INV.csv"
        diagram.findByType("outputfile", "out_altas").setPropertyValue("full_filename", altas_out)
        diagram.findByType("outputfile", "out_altas").run(None)

        # Iterate over target banking products and export stratified slices
        for product in products:
            diagram.findByType("outputfile", "out").setPropertyValue(
                "full_filename", f"{direc_out}{product}_{period}.csv"
            )
            diagram.findByType("derivenode", "COMPRA").setPropertyValue(
                "flag_expr", f"COMPRA_{product} = 1"
            )

            if product == "PAQUETE":
                condition = "TB_BLACK /= 1"
            elif product == "SC":
                condition = "PAQ_FLAG = 0 and TB_SC1 = 0"
            else:
                condition = "PENUMPER = PENUMPER"

            diagram.findByType("selectnode", "CONDICION").setPropertyValue("condition", condition)
            diagram.findByType("setglobals", "T/F COUNT").run(None)

            stream_globals = stream.getGlobalValues()
            t_count = stream_globals.getValue(modeler_api_module.GlobalValues.Type.SUM, "T_COUNT")
            f_count = max(t_count * 2, 150_000 - t_count)

            diagram.findByType("sample", "F_SAMPLER").setPropertyValue("fixed_counts", f_count)
            diagram.findByType("outputfile", "out").run(None)


if __name__ == "__main__":
    if "modeler" in globals():
        import modeler.api as _modeler_api  # type: ignore[import-not-found]

        run_slice_extraction(globals()["modeler"].script, _modeler_api)
    else:
        hist = periods_count(2006, 14)
        sample_src = (0, "CLP_NORM_T12", "CLP/")
        print("Dry-run lagged path resolution:", resolve_lagged_source_filename(sample_src, hist))
