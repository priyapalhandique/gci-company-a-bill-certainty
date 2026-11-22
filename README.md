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

The notebook downloads the v2 release asset `gci_company_a_bundle_v2.zip` and
verifies its SHA-256 before extraction. It then verifies `Client.csv` and
`Record.csv` against `source_hashes.json` before training.

Release asset SHA-256:
`64397993e675e1e14e5d492d91e7481520d08e98497d1a93debe2d393b50cb14`

v2 fixes a Colab pandas 3.x failure in the fairness audit: the old
`groupby(..., dropna=False)` call could raise
`ValueError: Categorical categories cannot be null` on string columns with
missing values. The audit now groups on an explicit object-cast label array and
keeps a separate `missing` group.

The data are redistributed only so the assignment notebook is reproducible in
Google Colab. They remain the property of GCI World / Company A and are not for
other use.
