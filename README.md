# Early Warning of Sovereign Debt Distress

Predicting sovereign debt distress within 12 months for 32 emerging-market economies using sequence models (GRU, Transformer, LSTM, BiLSTM) and tree-boosting baselines trained on macroeconomic, debt, balance-of-payments, commodity, and political-instability signals.

## Problem

Sovereign debt crises rarely emerge from a single variable. Debt rises while reserves shrink while global financing conditions tighten while political instability grows. Most existing early-warning models use a small set of macro indicators and treat each month independently, missing the cross-signal dynamics that build over time.

We frame the task as **binary time-series classification**: given 24 months of economic and political data for a country, predict whether a debt distress event will start within the next 12 months.

- **Panel:** 32 emerging-market economies × 300 months (Jan 2000 – Dec 2024) = 9,600 country-month observations  
- **Features:** 83 across 7 modalities — macro, debt, balance of payments, global conditions, commodity terms of trade, structural/history, political  
- **Label:** `distress_within_12m` — 1 if a distress episode begins within the next 12 months  
- **Class balance:** ~3–6% positive rate across splits

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── interim/                       # Interim-submission notebooks (early pipeline)
│   │   ├── 01_preprocessing.ipynb
│   │   └── 02_modeling.ipynb
│   ├── 01_data_cleaning.ipynb         # Final pipeline: raw + panel → train/val/test splits
│   └── 02_model_runner.ipynb          # Final pipeline: trains all models, evaluates, saves results
├── src/
│   └── data_cleaning_pipeline.py      # Raw archives → data/interim/ + data/processed/
├── data/
│   ├── raw/
│   │   ├── fred/                      # FRED global macro series (7 CSV files)
│   │   ├── ctot/                      # IMF Commodity Terms of Trade (1 CSV)
│   │   └── imf_dots/                  # IMF Direction of Trade Statistics (1 CSV)
│   ├── interim/                       # Cleaned single-source CSVs
│   ├── processed/
│   │   └── panel_full.csv             # Merged country-month base panel
│   └── final/                         # Train/val/test tables, sliding-window arrays, metadata
├── results/
│   ├── interim/                       # Interim-submission model results
│   └── final/                         # Final model results: summary, predictions, plots
├── dashboard/
│   └── sovereign_debt_dashboard.html  # Interactive results dashboard (open in browser)
├── figures/                           # Key figures referenced in this README
└── docs/
    ├── data_cleaning_methodology.docx
    └── modeling_guide.docx
```

## Data sources

The base panel (`data/processed/panel_full.csv`) was assembled from the following sources:

| Source | Variables | Coverage |
|--------|-----------|----------|
| IMF World Economic Outlook (WEO) | GDP growth, fiscal balance, current account, debt/GDP, inflation, unemployment | Annual, 2000–2024 |
| World Bank International Debt Statistics | External debt stock, debt service, short-term share | Annual, 2000–2024 |
| IMF Monetary and Financial Statistics | Policy rate, lending rate, deposit rate, real rate | Monthly, 2000–2024 |
| IMF Balance of Payments | FDI, portfolio flows, reserve assets, current account | Annual, 2000–2024 |
| BIS Debt Security Statistics | Government debt securities outstanding | Monthly, 2009–2024 |
| ACLED | Political violence events (pre-aggregated to country-month) | Monthly, 2000–2024 |
| Consumer Price Index (IMF IFS) | CPI level, MoM and YoY inflation | Monthly, 2000–2024 |
| IMF Exchange Rates (IFS) | Exchange rate level, momentum, volatility | Monthly, 2000–2024 |

The cleaning notebook (`notebooks/01_data_cleaning.ipynb`) adds FRED global factors and commodity terms of trade from:

| Source | Variables | Files in `data/raw/` |
|--------|-----------|----------------------|
| FRED — CBOE VIX | Global risk sentiment | `fred/VIXCLS.csv` |
| FRED — US 10Y Treasury | Global long rate | `fred/DGS10.csv` |
| FRED — Federal Funds Rate | US monetary policy | `fred/FEDFUNDS.csv` |
| FRED — Brent Crude | Oil price | `fred/DCOILBRENTEU.csv` |
| FRED — US CPI | US inflation (for real rate) | `fred/CPIAUCSL.csv` |
| FRED — Trade-Weighted USD (broad) | Dollar strength | `fred/DTWEXBGS.csv` |
| FRED — Trade-Weighted USD (legacy) | Dollar strength pre-2006 splice | `fred/TWEXBMTH.csv` |
| IMF Commodity Terms of Trade | Country-specific CoT index | `ctot/ctot_model_input_wide_2000_2024.csv` |
| IMF Direction of Trade (DOTS) | Monthly imports (reserves diagnostic only — excluded from model) | `imf_dots/imports_monthly.csv` |

## How to reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Option A — Run the model only (recommended)

All train/val/test splits are committed in `data/final/`. Clone the repo and run `notebooks/02_model_runner.ipynb` end to end.

**Path variables (Cell 1 of the notebook):**

```python
FINAL_DATA_DIR = Path("../data/final")    # folder containing the train/val/test splits
RESULT_DIR     = Path("../results/final") # folder where predictions, plots, and summary are written
```

Adjust these only if you are running from a directory other than `notebooks/`.

**What the notebook does, cell by cell:**
1. Set paths (`FINAL_DATA_DIR`, `RESULT_DIR`)
2. Install pinned dependencies (safe to skip if already installed)
3. Imports, config, and data loading — reads all flat CSV and NPZ files from `FINAL_DATA_DIR`
4. Scaling and shared inputs
5. Metrics, registration, and aligned ensembles
6. PyTorch utilities and losses
7. Model classes (GRU, LSTM, CNN, Transformer, Multimodal GRU-Transformer, etc.)
8. Train classical linear and tree baselines
9. Train regularised boosting models (XGBoost, LightGBM, CatBoost, Extra Trees, Random Forest)
10. Train flattened-window XGBoost
11. Train neural temporal models (CNN-1D, GRU, LSTM, TCN, Attention-GRU, Transformer, Multimodal GRU-Transformer)
12. Optional seed ensembles (disabled by default; set `RUN_OPTIONAL_SEED_ENSEMBLES = True` to enable)
13. Build top-only ensembles (neural rank ensemble, hybrid rank ensemble)
14. Final summary, plots, and export — writes `summary.csv` and per-model prediction CSVs to `RESULT_DIR`
15. Zip model results — writes `results/final_model_results.zip`
16–30. Visualisation cells — writes 12 plots to `RESULT_DIR/plots/`

**Outputs written to `results/final/`:**
- `summary.csv` / `summary_report_table.csv` — all model metrics
- `predictions/<model>__val.csv` and `predictions/<model>__test.csv` — per-row scores
- `plots/` — 12 plot files
- `../results/final_model_results.zip` — zip of the full results folder

**Runtime:** GPU (High-RAM) recommended. Expected: ~15 minutes on GPU, 60–90 minutes on CPU.

### 3. Option B — Rebuild the feature-engineering step

Run `notebooks/01_data_cleaning.ipynb` before the model runner. This notebook reads `data/processed/panel_full.csv` and the raw series in `data/raw/` and writes the final train/val/test splits to `data/final/`.

**Path variable (Cell 2 of the notebook):**

```python
DATA_DIR = Path("../data")
```

**Required input files:**
- `data/processed/panel_full.csv`
- `data/raw/fred/VIXCLS.csv`, `DGS10.csv`, `FEDFUNDS.csv`, `DCOILBRENTEU.csv`, `CPIAUCSL.csv`, `TWEXBMTH.csv`, `DTWEXBGS.csv`
- `data/raw/ctot/ctot_model_input_wide_2000_2024.csv`

**What the notebook does, cell by cell:**
1. Setup — imports
2. Set paths — set `DATA_DIR`
3. Validate required files — checks all inputs exist
4. Build curated distress labels — 22 hand-verified events → `data/interim/distress_events_curated.csv`
5. Default-history features — past-default counts and time-since-last-default
6. FRED global macro features — VIX, US 10Y, Fed Funds, Brent, USD index, real rate, etc.
7. Commodity Terms of Trade — country-specific CoT index, YoY and 6-month changes
8. Merge into final panel — joins panel, labels, default history, FRED, and CoT; applies time splits (train ≤ 2017, embargo 2018, val 2019–2020, embargo 2021, test ≥ 2022)
9. Save flat CSVs and windowed NPZ files — writes `train/val/test_flat.csv` and `train/val/test_windows.npz` to `data/final/`
10. Save metadata and summary statistics — writes `feature_metadata.json`, `class_distribution.json`, `summary_statistics.csv`
11. Zip outputs — writes `data/final_outputs.zip`

After running, proceed with Option A.

### 4. Option C — Full pipeline from raw archives

To rebuild `data/processed/panel_full.csv` from the original raw sources, follow the download instructions in [`data/README.md`](data/README.md) to populate `data/raw/`, then run:

```bash
python src/data_cleaning_pipeline.py
```

This produces the interim CSVs in `data/interim/` and the merged panel in `data/processed/panel_full.csv`. Then continue with Option B.

### 5. View the dashboard

Open `dashboard/sovereign_debt_dashboard.html` in any modern browser — no server needed. The dashboard shows model comparison, country risk timelines, score distributions, and precision-recall curves for all 18 models on the held-out test set (2022–2024).

## Time splits

| Split | Period | Rows | Positive rate |
|-------|--------|------|---------------|
| Train | Jan 2000 – Dec 2017 | 6,912 | 2.1% |
| Embargo | Jan 2018 – Dec 2018 | 384 | — |
| Validation | Jan 2019 – Dec 2020 | 768 | 6.3% |
| Embargo | Jan 2021 – Dec 2021 | 384 | — |
| Test | Jan 2022 – Dec 2024 | 1,152 | 4.4% |

Both embargo periods are excluded from training and evaluation to prevent leakage through the 24-month sliding windows at the split boundaries.

## Models

| Family | Models |
|--------|--------|
| Linear | Logistic Regression, Elastic-Net Logistic Regression |
| Tree boosting | XGBoost (tuned), LightGBM (tuned), CatBoost (tuned), Extra Trees, Random Forest |
| Recurrent | GRU, GRU seed ensemble, LSTM, Attention-GRU, Attention-GRU seed ensemble |
| Convolutional | 1D-CNN, TCN |
| Transformer | Multimodal GRU-Transformer, Multimodal GRU-Transformer seed ensemble, Transformer Encoder |
| Ensemble | Hybrid top-rank ensemble, Top neural rank ensemble |

All sequence models use 24-month sliding windows of shape `(B, 24, 83)`. Neural models are trained with focal loss (γ = 2.0), AdamW optimizer, ReduceLROnPlateau scheduler, and early stopping on validation AUPRC. Tree models use scale-pos-weight class weighting.

## Results

### Validation set (Jan 2019 – Dec 2020)

| Model | AUROC | AUPRC |
|-------|------:|------:|
| Hybrid rank ensemble | 0.827 | 0.451 |
| GRU | 0.846 | 0.422 |
| GRU seed ensemble | 0.827 | 0.392 |
| Multimodal GRU-Transformer (seed ens.) | 0.808 | 0.484 |
| Top neural rank ensemble | 0.819 | 0.405 |
| Transformer Encoder | 0.824 | 0.327 |
| Logistic Regression | 0.833 | 0.313 |
| Elastic-Net Logistic Regression | 0.832 | 0.328 |
| Attention-GRU | 0.800 | 0.358 |
| LSTM | 0.718 | 0.296 |

Full results for all 18 models (val + test, multiple thresholds) are in `results/final/summary.csv`.

### Test set (Jan 2022 – Dec 2024)

The test set covers the post-COVID stress period including Sri Lanka (2022), Ghana (2022), Ethiopia (2023), and Kenya (2024). The **Multimodal GRU-Transformer** achieves the highest test AUROC (0.924) and AUPRC (0.275). The **hybrid rank ensemble** achieves the best balanced test AUROC (0.775) with stable generalisation from validation.

![Model comparison](figures/model_comparison_final.png)

![PR curves](figures/pr_curves_final.png)

![Lead time](figures/lead_time_final.png)

## References

[1] D. Beers, E. Jones, and J. Walsh, “BoC-BoE Sovereign Default Database: What’s New in 2023?,” Bank of Canada Staff Analytical Note, 2023.

[2] C. M. Reinhart and K. S. Rogoff, This Time Is Different: Eight Centuries of Financial Folly. Princeton University Press, 2009.

[3] G. L. Kaminsky, S. Lizondo, and C. M. Reinhart, “Leading Indicators of Currency Crises,” IMF Staff Papers, vol. 45, no. 1, pp. 1–48, 1998.

[4] P. Manasse and N. Roubini, “Rules of Thumb for Sovereign Debt Crises,” Journal of International Economics, vol. 78, no. 2, pp. 192–205, 2009.

[5] M. Dawood, N. Horsewood, and F. Strobel, “Predicting Sovereign Debt Crises: An Early Warning System Approach,” Journal of Financial Stability, vol. 28, pp. 16–28, 2017.

[6] A. Petropoulos, V. Siakoulis, E. Stavroulakis, and D. Pinotsis, “Towards an Early Warning System for Sovereign Defaults Leveraging on ML Methodologies,” Intelligent Systems in Accounting, Finance and Management, vol. 29, no. 2, pp. 118–129, 2022.

[7] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016.

[8] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[9] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, “CatBoost: Unbiased Boosting with Categorical Features,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[10] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal Loss for Dense Object Detection,” in Proceedings of the IEEE International Conference on Computer Vision, pp. 2980–2988, 2017.

[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention Is All You Need,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[12] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[13] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2014.

[14] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[15] C. Raleigh, A. Linke, H. Hegre, and J. Karlsen, “Introducing ACLED: An Armed Conflict Location and Event Dataset,” Journal of Peace Research, vol. 47, no. 5, pp. 651–660, 2010.

[16] International Monetary Fund, “World Economic Outlook, International Financial Statistics, and Balance of Payments Data.” Available at: https://www.imf.org/en/Data.

[17] World Bank, “International Debt Statistics.” Available at: https://www.worldbank.org/en/programs/debt-statistics/ids.

[18] Bank for International Settlements, “Debt Securities Statistics.” Available at: https://data.bis.org/topics/DSS.

[19] Federal Reserve Bank of St. Louis, “Federal Reserve Economic Data.” Available at: https://fred.stlouisfed.org/.

[20] International Monetary Fund, “Commodity Terms of Trade Database.” Available at: https://data.imf.org/.

[21] T. Kristof, “Sovereign Default Forecasting in the Era of the COVID-19 Crisis,” Journal of Risk and Financial Management, vol. 14, no. 10, p. 494, 2021.

[22] S. Barth´elemy, L. Music, A. Gautier, and A. Vouldis, “Early Warning System for Currency Crises Using Long Short-Term Memory and Gated Recurrent Unit Neural Networks,” Journal of Forecasting, vol. 43, no. 4, pp. 1142–1165, 2024.

[23] H. Emami, C. Zhang, and O. R. Za¨ıane, “Modality-Aware Transformer for Financial Time Series Forecasting,” arXiv preprint arXiv:2310.01232, 2024.

[24] C. Reimann, “Predicting Financial Crises: An Evaluation of ML Algorithms and Model Explainability for Early Warning Systems,” Review of Evolutionary Political Economy, vol. 5, pp. 283–310, 2024.

[25] L. Laeven and F. Valencia, “Systemic Banking Crises Database II,” IMF Economic Review, vol. 68, no. 2, pp. 307–361, 2020.

[26] K. Bluwstein, M. Buckmann, A. Joseph, S. Kang, S. Kapadia, and O. Simsek, “Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach,” Bank of England Staff Working Paper, 2020.
