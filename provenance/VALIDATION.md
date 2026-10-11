# Validation performed while preparing this snapshot

- All 56 archived Python/notebook sources were parsed successfully.
- Archived `.py` source files are byte-identical to their original local files.
- Notebook executable code and markdown were preserved; only outputs and
  execution metadata were removed.
- Potential embedded credentials were screened before packaging. Authentication
  files, raw waveform archives, and manuscript documents were excluded.
- The 44 compact label files preserve the original `y_pred_last.npy` members
  byte-for-byte. Their SHA-256 checksums are in `derived_label_manifest.json`.
- `scripts/run_pcs_null.py --observed-only` verified all observed selections
  against the original workbook and reproduced **35/44** and enrichment
  **3.307086614173** in the current Python 3.9 analysis environment.
- A complete low-cost execution check used two shifts/event, 100 global
  replicates, seed 20260812. It processed all 44 events, checked observed
  selections, and wrote numerical tables and PNG/PDF/SVG figures successfully.
  Multiprocessing was restricted in the validation sandbox; the original
  implementation fell back to serial execution.
- The low-cost check is an execution test, not a replacement for the manuscript
  null run or a replication of its P value.
- Full DEC retraining, all physical-feature calculations, rainfall acquisition,
  all manuscript figures, and operational warning performance were not tested
  during packaging.
