# Debris-flow seismic precursors

Source code supporting **Deep Unsupervised Learning Reveals Recurrent Candidate
Seismic Precursors to Debris-Flow Initiation**.

The study analyzes 24-hour seismic windows preceding the recorded onset of
44 debris-flow events in Illgraben, Switzerland. It is a retrospective
candidate-precursor study.

## Contents and provenance

- `original_code/uilt_oneday_all/`: source scripts from the research project,
  organized in their original relative directory structure. Python scripts are
  byte-identical copies. Notebook code and markdown are preserved; outputs,
  execution counts, and execution metadata are removed.
- `original_code/data/降雨数据下载/`: the original GPM IMERG acquisition script;
  credential/configuration files are excluded.
- `scripts/run_pcs_null.py`: a portable entry point that redirects paths and
  calls the original PCS/null-test functions without changing their algorithms.
- `derived_data/dec_labels/`: 44 compact NPZ caches containing **only** the
  original `y_pred_last` labels. They are not complete DEC output archives.
- `derived_data/precursor_clusters_all_metrics.xlsx`: the original reference
  workbook for checking observed PCS selections.
- `reference_results/`: existing numerical reference outputs, not results
  invented or recomputed during code packaging.
- `provenance/`: source and packaged SHA-256 checksums and label extraction
  records. These establish the files included in this snapshot, not the exact
  historical software state of every research run.
- `docs/WORKFLOW.md`: manuscript analysis chain, input requirements, and
  interpretation boundaries.

No manuscript documents, account credentials, raw SAC/miniSEED records,
spectrogram tensors, or trained model weights are bundled.

## Quick reproduction of observed PCS results

Use Python 3.9 in a separate environment and install the analysis dependencies:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/run_pcs_null.py --observed-only
```

Expected observed results from the manuscript dataset:

```text
PCS selected events: 35/44
Median final-hour enrichment: 3.307086614173
```

For a low-cost execution check of the complete null-test workflow:

```bash
python scripts/run_pcs_null.py --n-shifts 2 --global-replicates 100 --workers 1
```

This deliberately small run is **not** an inferential reproduction of the
reported null distribution. The manuscript reference run uses:

```bash
python scripts/run_pcs_null.py --n-shifts 120 --global-replicates 20000 --seed 20260812 --workers 4
```

The saved reference summary reports a null median terminal enrichment of
1.474060675801. Runtime depends on hardware. Results and plots are written to
`outputs/pcs_null/`. Cross-version floating-point differences can affect near-ties
and exact null results; compare against the committed reference tables.

## Which PCS implementation is authoritative?

The manuscript's primary 24-hour cohort uses
`original_code/uilt_oneday_all/03visual/09_export_precursor_curve_metrics.py`.
The hard criteria are `z_final > 2`, `c_value < 1`, and
`terminal_growth_flag == 1`. The terminal window is **1 hour** and the preceding
comparison window is **5 hours**. If a state occurs in the preceding 5 hours,
its final-hour rate must be at least half that earlier rate. If it has no
preceding-window occurrences, at least one final-hour occurrence is required.

The selected cluster is ranked using within-event standardized quantities:

```text
PCS = Z(z_final) + Z(rate_acceleration) - Z(c_value) + Z(terminal_growth_ratio)
```

The later `09_psc.py` two-hour plotting variant and the fixed-K global DEC
experiment are not the primary manuscript PCS implementation and are not included.
Global autoencoder scripts are included only because the shorter-window
reclustering chain depends on them; this is documented separately.

## Full training and figure reproduction

The bundled labels suffice for the observed PCS and circular-shift null
analyses. They do **not** suffice for retraining DEC, reconstructing all figures,
computing physical features from waveforms, or redoing the rainfall analysis.
Those stages require additional source data and intermediate outputs listed in
`docs/WORKFLOW.md`.

Original scripts retain author-specific Windows paths, working-directory
assumptions, and event selections. Configure these before execution. Do not run
all scripts indiscriminately: some are helpers or plotting entry points, and
some original scripts execute immediately on import. Read the workflow guide.

The training and analysis environments are distinct:

- `requirements-analysis.txt`: confirmed currently imported analysis versions.
- `requirements-patterns.txt`: additional local distributions for pattern analysis.
- `requirements-training.txt`: distribution versions present in the legacy
  Python 3.7.16 DEC environment; this is not an end-to-end training validation.

Raw seismic data are associated with network XP:
https://doi.org/10.12686/sed/networks/xp.
Station metadata, event windows, and preprocessing choices must match the study.
GPM IMERG data require their own provenance and access arrangements; no Earthdata
credentials are included.



## Citation and archiving

The manuscript is not yet assigned a publication DOI. Cite this repository with
the exact commit used for review, then update the citation after publication.
An immutable archival deposit and software DOI have not been created by this
upload. A GitHub URL alone should not be described as a preservation archive.

No new blanket license is assigned in this snapshot. Existing attribution and
third-party references in the original source are retained. The authors should
confirm software licensing before granting redistribution rights.

Contact: Zhiyong Huang, huangzhiyong@imde.ac.cn.
