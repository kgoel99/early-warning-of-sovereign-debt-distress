# Data

The cleaned, model-ready dataset ships in this repository under `data/interim/`, `data/processed/`, and `data/final/`. Only `data/raw/` (the original provider archives) is excluded from version control because of file-size and licensing considerations; raw inputs can be downloaded directly from the original public providers using the links below.

## Folder layout

```
data/
├── raw/         # original provider archives — not committed; download separately (see below)
├── interim/     # cleaned single-source files (committed) — produced by src/data_cleaning_pipeline.py
├── processed/   # merged country-month panel (committed) — panel_full.csv
└── final/       # train/val/test flat tables, 24-month sliding-window .npz arrays, metadata, results (committed)
```

## What's already in this repo

| Folder | Contents | Size |
|---|---|---|
| `data/interim/` | 13 single-source files (ACLED aggregates, BIS, BOP + audit, CPI, distress events, external debt, FX, interest rates, labels, political violence, WEO) | ~5.8 MB |
| `data/processed/` | `panel_full.csv` — 9,600 rows × 87 columns | ~8.2 MB |
| `data/final/` | `train_flat.csv`, `val_flat.csv`, `test_flat.csv`, matching `_meta.csv` files, `train_windows.npz`, `val_windows.npz`, `test_windows.npz`, `panel_full.csv` (96 cols, with engineered features), `feature_metadata.json`, `class_distribution.json`, `summary_statistics.csv`, and the canonical `results_improved_baselines_v2_all_models/` model outputs | ~23 MB |

The `notebooks/02_modeling.ipynb` notebook can be run directly against these files — no extra download step is needed.

A bundled archive (`early-warning-data-v1.0.zip`, ~9 MB) of the same data is also attached to the [v1.0 Release](https://github.com/kgoel99/early-warning-of-sovereign-debt-distress/releases/tag/v1.0) for users who want a single-file download.

## Rebuilding from raw sources

To reproduce the pipeline end to end, place the provider archives below into `data/raw/` and run `python src/data_cleaning_pipeline.py` from the project root. The pipeline regenerates `data/interim/` and `data/processed/`; `notebooks/01_preprocessing.ipynb` then regenerates `data/final/`.

| Source | Provider | What we use | Access |
|---|---|---|---|
| IMF World Economic Outlook (WEO) | International Monetary Fund | Annual macro indicators (GDP, inflation, fiscal balance, etc.) repeated to monthly frequency | https://www.imf.org/en/Publications/WEO |
| IMF Balance of Payments (BOP) | International Monetary Fund | Current account, reserves, FDI, portfolio flows, remittances (monthly where available) | https://data.imf.org/?sk=7A51304B-6426-40C0-83DD-CA473CA1FD52 |
| World Bank International Debt Statistics (IDS) | World Bank | External debt levels, debt service ratio, debt structure | https://data.worldbank.org/products/ids |
| BIS Government Debt Securities | Bank for International Settlements | Outstanding government debt securities, quarterly | https://www.bis.org/statistics/secstats.htm |
| ACLED conflict events | Armed Conflict Location & Event Data Project | Country-month aggregates of battles, protests, riots, fatalities, civilian targeting | https://acleddata.com (free academic access; redistribution restricted) |
| BoC–BoE Sovereign Default Database | Bank of Canada / Bank of England | Distress episodes used to construct the `distress_within_12m` label | https://www.bankofcanada.ca/?p=224812 |
| Foreign exchange and CPI | IMF Data Portal / national statistics | Monthly FX rates and consumer price indices | https://data.imf.org |

> **ACLED note:** ACLED requires a free academic registration to download the original event-level data. The files committed in this repository are derived country-month aggregates, not the raw event records.

## File names expected by the cleaning pipeline

`src/data_cleaning_pipeline.py` expects the raw archives to be present in `data/raw/` with the original provider filenames (IMF WEO Excel exports, World Bank IDS CSVs, ACLED country exports, BIS quarterly CSVs, IMF BOP CSVs, BoC–BoE Excel files). See the file headers and parsing functions in the script for the exact filenames recognised.

## Coverage caveats

- BIS government debt securities cover 17 of the 32 panel countries.
- ACLED event-type detail covers 28 of 32; the pipeline drops features that exceed 50% missingness, leaving the aggregated political-violence signal as the stable political feature.
- T-bill interest rates are available for ~23 of 32 countries; the same sparsity rule applies.

## Country list (32)

Argentina, Brazil, Colombia, Dominican Republic, Ecuador, Egypt, Ethiopia, Ghana, Hungary, India, Indonesia, Jamaica, Jordan, Kenya, Lebanon, Malaysia, Mexico, Mozambique, Nigeria, Pakistan, Peru, Philippines, Poland, Romania, Russia, South Africa, Sri Lanka, Tunisia, Turkey, Ukraine, Venezuela, Zambia.
