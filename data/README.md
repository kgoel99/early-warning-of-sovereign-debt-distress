# Data

This directory is intentionally empty in version control. The cleaned, model-ready dataset is distributed as a GitHub Release asset, and the raw inputs are downloaded from the original public providers (links below).

## Folder layout

```
data/
├── raw/         # original archives downloaded from each source
├── interim/     # single-source cleaned files produced by src/data_cleaning_pipeline.py
├── processed/   # merged country-month panel (panel_full.csv)
└── final/       # train/val/test flat tables, 24-month sliding-window .npz arrays, metadata
```

## Option 1 — download the cleaned dataset (fastest)

The cleaned panel, train/val/test splits, sliding-window arrays, and feature metadata are bundled and attached to the [v1.0 Release](https://github.com/kgoel99/early-warning-of-sovereign-debt-distress/releases/tag/v1.0).

Direct download: [`early-warning-data-v1.0.zip`](https://github.com/kgoel99/early-warning-of-sovereign-debt-distress/releases/download/v1.0/early-warning-data-v1.0.zip) (~9 MB). Unzip into the project root so the contents land under `data/interim/`, `data/processed/`, and `data/final/`. After unzipping, `notebooks/02_modeling.ipynb` runs end to end.

## Option 2 — rebuild from raw sources

Place the raw archives below into `data/raw/` and run `python src/data_cleaning_pipeline.py` from the project root. The pipeline writes `data/interim/` and `data/processed/`; `notebooks/01_preprocessing.ipynb` then writes `data/final/`.

| Source | Provider | What we use | Access |
|---|---|---|---|
| IMF World Economic Outlook (WEO) | International Monetary Fund | Annual macro indicators (GDP, inflation, fiscal balance, etc.) repeated to monthly frequency | https://www.imf.org/en/Publications/WEO |
| IMF Balance of Payments (BOP) | International Monetary Fund | Current account, reserves, FDI, portfolio flows, remittances (monthly where available) | https://data.imf.org/?sk=7A51304B-6426-40C0-83DD-CA473CA1FD52 |
| World Bank International Debt Statistics (IDS) | World Bank | External debt levels, debt service ratio, debt structure | https://data.worldbank.org/products/ids |
| BIS Government Debt Securities | Bank for International Settlements | Outstanding government debt securities, quarterly | https://www.bis.org/statistics/secstats.htm |
| ACLED conflict events | Armed Conflict Location & Event Data Project | Country-month aggregates of battles, protests, riots, fatalities, civilian targeting | https://acleddata.com (free academic access; redistribution restricted) |
| BoC–BoE Sovereign Default Database | Bank of Canada / Bank of England | Distress episodes used to construct the `distress_within_12m` label | https://www.bankofcanada.ca/?p=224812 |
| Foreign exchange and CPI | IMF Data Portal / national statistics | Monthly FX rates and consumer price indices | https://data.imf.org |

> **ACLED note:** ACLED requires a free academic registration to download conflict-event data, and their license restricts redistribution. The cleaned, country-month *aggregates* used here are derived statistics, but please consult the ACLED Terms of Use before publishing any derivative work.

## File names expected by the cleaning pipeline

`src/data_cleaning_pipeline.py` expects the raw archives to be present in `data/raw/` with the original provider filenames (IMF WEO Excel exports, World Bank IDS CSVs, ACLED country exports, BIS quarterly CSVs, IMF BOP CSVs, BoC–BoE Excel files). See the file headers and parsing functions in the script for the exact filenames recognised.

## Coverage caveats

- BIS government debt securities cover 17 of the 32 panel countries.
- ACLED event-type detail covers 28 of 32; the pipeline drops features that exceed 50% missingness, leaving the aggregated political-violence signal as the stable political feature.
- T-bill interest rates are available for ~23 of 32 countries; the same sparsity rule applies.

## Country list (32)

Argentina, Brazil, Colombia, Dominican Republic, Ecuador, Egypt, Ethiopia, Ghana, Hungary, India, Indonesia, Jamaica, Jordan, Kenya, Lebanon, Malaysia, Mexico, Mozambique, Nigeria, Pakistan, Peru, Philippines, Poland, Romania, Russia, South Africa, Sri Lanka, Tunisia, Turkey, Ukraine, Venezuela, Zambia.
