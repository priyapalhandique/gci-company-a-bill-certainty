# GCI World 2026 - Company A reproducibility bundle

This repository contains the data and source modules used by the final-assignment
notebook `PriyapalDSAI.ipynb` (Company A / "Bill Certainty" proposal).

## Contents

- `data/Client.csv`, `data/Record.csv` - the two course-provided tables
  (100,000 customers each).
- `src/config.py`, `src/guardrails.py`, `src/phase2_models.py` - portable build
  modules used by the notebook.
- `source_hashes.json` - SHA-256 manifest recorded from the original course
  files.

## Reproducibility

The notebook downloads the release asset `gci_company_a_bundle.zip` and verifies
its SHA-256 before extraction. It then verifies `Client.csv` and `Record.csv`
against `source_hashes.json` before training.

The data are redistributed only so the assignment notebook is reproducible in
Google Colab. They remain the property of GCI World / Company A and are not for
other use.
