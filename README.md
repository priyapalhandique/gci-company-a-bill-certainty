# GCI World 2026 — Company A reproducibility bundle

This repository contains the data, source modules, and executed notebook for the
GCI World 2026 final assignment (Company A / "Bill Certainty" proposal).

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyapalhandique/gci-company-a-bill-certainty/blob/main/PriyapalDSAI.ipynb)

Choose **Runtime → Run all**. The notebook downloads its SHA-256-verified data
and code bundle on the first code cell; no Drive mount or local setup is needed.

## Contents

- `PriyapalDSAI.ipynb` — the executed submission notebook (60 cells, 21 code
  cells, zero errors, `GUARDRAIL CHECK: PASS`).
- `data/Client.csv`, `data/Record.csv` — the two course-provided tables
  (100,000 customers each).
- `src/config.py`, `src/guardrails.py`, `src/phase2_models.py` — portable build
  modules used by the notebook.
- `source_hashes.json` — SHA-256 manifest recorded from the original course
  files.

## Reproducibility

The notebook downloads the release asset `gci_company_a_bundle.zip` and verifies
its SHA-256 before extraction. It then verifies `Client.csv` and `Record.csv`
against `source_hashes.json` before training.

Release asset SHA-256:
`d595cc405cdee698a12cffe68c325842da18dfb16836f1b90da3d3ec42999e46`

The data are redistributed only so the assignment notebook is reproducible in
Google Colab. They remain the property of GCI World / Company A and are not for
other use.
