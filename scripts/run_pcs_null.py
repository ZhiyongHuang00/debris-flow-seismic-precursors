"""Portable entry point for the unmodified manuscript PCS/null implementation.

Only input/output paths are redirected. The scientific scoring, selection,
circular-shift sampling, and Monte Carlo resampling use the original functions.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "original_code" / "uilt_oneday_all"
SOURCE_SCRIPT = SOURCE_ROOT / "uilt_paper_figure" / "011_PSC_test.py"

# Execute on module import as well, so Windows multiprocessing child processes
# register the same source module and path configuration before unpickling.
spec = importlib.util.spec_from_file_location("manuscript_pcs_null", SOURCE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load manuscript source: {SOURCE_SCRIPT}")
null = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = null
spec.loader.exec_module(null)
null.PCS_REFERENCE_SCRIPT = SOURCE_ROOT / "03visual" / "09_export_precursor_curve_metrics.py"
null.DEC_LABEL_DIR = ROOT / "derived_data" / "dec_labels"
null.AUTHORITATIVE_PCS_WORKBOOK = ROOT / "derived_data" / "precursor_clusters_all_metrics.xlsx"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-shifts", type=int, default=null.DEFAULT_N_SHIFTS)
    parser.add_argument("--global-replicates", type=int, default=null.DEFAULT_GLOBAL_REPLICATES)
    parser.add_argument("--seed", type=int, default=null.DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pcs_null")
    parser.add_argument("--observed-only", action="store_true", help="Verify the 44 observed selections without null sampling or plotting.")
    args = parser.parse_args()
    if args.n_shifts < 1 or args.global_replicates < 1 or args.workers < 1:
        parser.error("shifts, replicates, and workers must be positive")
    if args.observed_only:
        import numpy as np
        pcs = null.load_pcs_module()
        rows = []
        for event, path in null.discover_event_files():
            with np.load(path, allow_pickle=False) as archive:
                labels = np.asarray(archive["y_pred_last"], dtype=int)
            rows.append({"event": event, "n_samples": len(labels), "observed": null.selected_state_result(labels, event, pcs)})
        null.validate_observed_against_workbook(rows)
        selected = [row["observed"] for row in rows if row["observed"]["selected"]]
        print(f"PCS selected events: {len(selected)}/{len(rows)}")
        print(f"Median final-hour enrichment: {np.median([row['terminal_enrichment'] for row in selected]):.12f}")
        return
    output = args.output_dir.resolve()
    results = null.run_event_analyses(null.discover_event_files(), args.n_shifts, args.seed, args.workers)
    null.validate_observed_against_workbook(results)
    observed = null.observed_summary(results)
    resampled = null.global_null_resampling(results, args.global_replicates, args.seed)
    tables = null.build_output_tables(results, resampled, observed, args.n_shifts, args.seed)
    null.save_source_data(output, tables, results, resampled, observed, args)
    null.plot_publication_figure(output, results, resampled, observed)
    print(tables[0].to_string(index=False))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
