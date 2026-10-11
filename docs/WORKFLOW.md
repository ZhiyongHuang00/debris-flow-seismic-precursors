# Manuscript workflow and reproduction boundaries

All source paths below are relative to `original_code/uilt_oneday_all/`.
This is a current source snapshot with recorded hashes. Run logs do not establish
an immutable historical source version for every intermediate result.

## 1. Seismic acquisition and preprocessing

Source: `01data_preprocess/01_download_ILLdata.py`,
`01_down_self_time.py`, `03_cut_one_minutes_event.py`, and `04_make_stft.py`.

Inputs: waveform records, appropriate instrument-response metadata, and the
event inventory originally named `data/新版完整数据集时间.xlsx`.
The cutter currently contains the explicit row slice `[34:35]`, reflecting an
event-specific execution. Configure the intended events and station response
before regenerating a complete dataset. Preserve and record the UTC window
convention; do not infer an unobserved physical initiation time from the label.

`04_make_stft.py` uses `fs=100`, `nperseg=19`, `nfft=127`, and SciPy's default
Hann window/overlap, then cubic interpolation to 512 time bins and min–max
normalization. These exact operations are preserved, not replaced with a newly
designed preprocessing pipeline.

Outputs: one-minute SAC segments and `data/STFT_oneday_all/Z/*.npz` archives
with `stft` and `files` arrays. The raw data and STFT archives are not bundled.

## 2. Event-specific primary DEC

Run from `02clustering/` after configuring absolute paths:

1. `01_Autodecoder.py`: per-event autoencoder pretraining.
2. `02_calculate_Kmeans_metrics.py`: per-event cluster diagnostics.
3. `03_find_Kmeans_metrics.py`: knee-based K selection and `summary_best_k.xlsx`.
4. `01_DEC.py`: per-event DEC using the selected K and saved pretraining weights.

Local dependencies: `Model_AE.py`, `metrics.py`, `plot_distribution.py`.
`Model_AE.py` is the layer-normalized architecture, not the alternate
`Model_60s.py` architecture.

Outputs under `02clustering/result/`: pretraining weights/logs, K diagnostics,
DEC losses, and `DEC_result/all_parement/event*_all_parement.npz`.
The complete NPZ files contain spectrograms, reconstruction arrays, latent
features, labels, probabilities, file references, and centroid distances.
The **compact bundled caches contain labels only** and cannot substitute for
these complete archives when a script requests other keys.

## 3. Primary PCS and selection-aware null

Primary scorer: `03visual/09_export_precursor_curve_metrics.py`.
Null implementation: `uilt_paper_figure/011_PSC_test.py`.
Main all-event plotting: `uilt_paper_figure/012_all_event_and_test.py` and notebook.

Use the repository-root `scripts/run_pcs_null.py` for portable reproduction.
It loads the exact archived source modules and redirects only their input paths.
The scorer compares cumulative state occupancy against a background baseline,
fits nonlinear count evolution, applies the terminal-activity rule, and ranks
eligible states within each event.

Null test: preserve each label sequence, sample stratified nonzero circular
shifts, rerun all metrics and selection, and resample event-level null results.
Saved run parameters: 120 shifts/event, 20,000 global replicates, seed 20260812.
The reference summary distinguishes two statistics: selected-event count
(35 versus null median 32; P approximately 0.150) and median enrichment
(3.307 versus 1.474; P approximately 0.0000500). Do not conflate these tests.

## 4. Independent shorter-window clustering and alignment

Training dependencies: `02clustering/05_Autodecoder_all_event.py`,
`06_all_kneed_kmeans.py`, `08_DEC_different_duration_kneed.py`, and
`09_DEC_different_duration.py`.
These use global pretraining weights followed by event/window-specific DEC.

Analysis: `03visual/compared_different_duration/02_PCS_different_dura.py`
and `06_match_24h.py`; related overlap and comparison helpers are included.
Final reference tracking: `uilt_paper_figure/08_different_duration_base24.py`
and `Figure_06_difference_duration.ipynb`.

The legacy duration scorer points its original 24-hour reference at a
fixed-K/global-results folder. The final base24 plotting/analysis script
explicitly ignores that legacy reference and re-anchors to the primary
event-specific 35-event PCS cohort. Do not use the legacy reference as the
primary manuscript result. Short-window NPZ archives, lookup tables, and
alignment workbooks are not bundled.

## 5. Physical features and the two patterns

Feature computation: `calculate_paramater/02_calculate_all.py` and
`Polarization.py`. The former retains embedded third-party attribution for
peak-detection code. Preserve and review that attribution when licensing.

Matching: `03visual/二次聚类/00_matching_feature.py` uses original feature CSVs,
complete DEC archives, and the precursor-label spreadsheet. Verify that the
labels match the authoritative primary PCS cohort before running.

Second-stage clustering: `03visual/二次聚类/08_kmeans_precursor2.py`.
Physical/spectral summaries: `09_kmeans2_mode_waveform_process.py`, which imports
`09_kmeans_secondary_cluster_physical_process_evidence.py` as a helper.
Main plotting: `uilt_paper_figure/05_new_ record-section.py`,
`05_two_model.ipynb`, and `figure_rainfall_couple_event.py`.

This stage requires raw three-component waveforms, feature tables, complete
DEC archives, and the matched-feature workbook. They are not bundled.
The committed `reference_results/kmeans_clustering_results.csv` is a small
derived reference table, not the complete input feature matrix.

Frequency bins and physical frequencies must remain distinct: 100 Hz waveform
sampling has a 50 Hz Nyquist limit. Source packaging does not correct existing
feature-unit ambiguities or rerun this stage.

## 6. Rainfall context

Processing: `03visual/rain_coupling/01_rainfall_couple.py`.
Duration comparison: `03visual/compared_different_duration/04_different_model_rainfall.py`.
Final analysis and plotting: `uilt_paper_figure/010_rainfall_coupling.py` and
`Figure_08_rainfall_couple.ipynb`.

Inputs include monthly GPM IMERG CSVs, the processed continuous rainfall series,
the event rainfall workbook, primary PCS workbook, and pattern/event table.
Monthly CSVs were produced by a separate Earthdata acquisition script that is
not bundled here. Authentication configuration must never be committed.
The data products and acquisition provenance must be archived separately.

## 7. Figure sources

| Manuscript content | Main source entry points |
| --- | --- |
| Study region and example waveform/spectral records | `uilt_paper_figure/map.py`, `MAP.ipynb`, `01_Figure1.py`, `01_waveform_spectrum.ipynb` |
| Model architecture and clustering diagnostics | `02_model_框架_公式更新版.py`, `02_model_result.py`, `02_model_result_figure.ipynb`, `02_cluster_results_DEC_quality.ipynb` |
| Event-scale PCS example and cluster diagnostics | `03_event_scale_precursor_cluster.ipynb`, `04_event_cluster_diagnostics.ipynb` |
| All-event precursor organization and null tests | `011_PSC_test.py`, `012_all_event_and_test.py`, `012_all_event_and_test.ipynb` |
| Cross-window reference-state recovery | `08_different_duration_base24.py`, `Figure_06_difference_duration.ipynb` |
| Two temporal–spectral patterns | `05_new_ record-section.py`, `05_two_model.ipynb` |
| Rainfall comparisons | `010_rainfall_coupling.py`, `Figure_08_rainfall_couple.ipynb` |
| Supporting rainfall, PCS sensitivity, and secondary clustering | `Figure_S01_rainfall.ipynb`, `Figure_S5_PCS参数及启动时刻稳健性图.ipynb`, `FigureS6_second_clustering.ipynb` |

Figure numbers and several folder names changed during manuscript preparation.
The table maps content rather than asserting historical file names establish
the exact final image version. Cartopy, PyGMT/GMT, spatial datasets, and other
map resources are additional optional dependencies not part of the PCS lock.

## Before claiming full reproducibility

- Confirm event-window definitions and potential overlaps.
- Archive the event catalog, rainfall data, required derived tables and model
  artifacts with their reuse permissions and identifiers.
- Verify parameter counts and frequency-unit interpretation against the code.
- Run the full training/feature/rainfall/figure chain in documented environments.
- Obtain an immutable software archive/version DOI and confirm licensing.

Uploading this code snapshot alone is not certification that all of those
scientific and preservation requirements have been completed.
