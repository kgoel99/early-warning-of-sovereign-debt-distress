
"""
sovereign_data_cleaning_pipeline.py

Consolidated, heavily documented pipeline for cleaning and arranging the
sovereign debt distress early-warning datasets that were packaged in:

- Raw Data-20260403T035740Z-1-001.zip
- Large Raw Data-20260403T040046Z-1-001.zip

This script is written as a reproducible end-to-end pipeline. It follows the
project's cleaning specification closely and is structured so teammates can
inspect or modify each step independently.

What this script does
---------------------
1. Extract raw archives.
2. Build country-level monthly features from each source.
3. Build forward-looking default/distress labels.
4. Merge into a complete country-month panel for 2000-01 to 2024-12.
5. Apply feature filtering, missingness handling, and additional engineering.
6. Save:
   - interim cleaned source files
   - merged panel
   - flat train/val/test CSVs
   - 24-month sequence windows for LSTM / Transformer models
   - metadata reports

Important implementation choices
--------------------------------
- Annual data are expanded to all months of the same year. They are not
  linearly interpolated because these are reported annual statistics.
- IMF and BOP files are wide-format exports with time columns embedded as
  headers. The script identifies date-like columns using regular expressions.
- BoC-BoE files contain metadata, series dictionaries, and an OBSERVATIONS
  block rather than a standard CSV from row 1. The script explicitly searches
  for the OBSERVATIONS section and parses only that block.
- Optional sources (especially BIS debt securities) are wrapped in
  try/except blocks so the full pipeline can still run if a complex file
  fails to parse cleanly.
- The pipeline is written with pandas only. It is intentionally readable and
  traceable rather than micro-optimized.

Typical usage
-------------
python sovereign_data_cleaning_pipeline.py \
  --raw-zip "/path/to/Raw Data-20260403T035740Z-1-001.zip" \
  --bop-zip "/path/to/Large Raw Data-20260403T040046Z-1-001.zip" \
  --output-dir "/path/to/output_root"

Outputs are written to:
output_root/
  data/interim/
  data/processed/

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

COUNTRIES = {
    "Argentina": "ARG", "Brazil": "BRA", "Ecuador": "ECU", "Colombia": "COL",
    "Mexico": "MEX", "Peru": "PER", "Venezuela": "VEN",
    "Ghana": "GHA", "Kenya": "KEN", "Nigeria": "NGA", "South Africa": "ZAF",
    "Zambia": "ZMB", "Ethiopia": "ETH", "Mozambique": "MOZ",
    "India": "IND", "Indonesia": "IDN", "Malaysia": "MYS", "Pakistan": "PAK",
    "Sri Lanka": "LKA", "Philippines": "PHL",
    "Egypt": "EGY", "Lebanon": "LBN", "Tunisia": "TUN", "Jordan": "JOR",
    "Turkey": "TUR", "Ukraine": "UKR", "Russia": "RUS",
    "Poland": "POL", "Hungary": "HUN", "Romania": "ROU",
    "Dominican Republic": "DOM", "Jamaica": "JAM"
}
ISO3_LIST = sorted(COUNTRIES.values())
CANONICAL_BY_ISO3 = {v: k for k, v in COUNTRIES.items()}

COUNTRY_VARIANTS = {
    # canonical + common IMF / WB / ACLED variants
    "ARGENTINA": "ARG",
    "BRAZIL": "BRA",
    "BRASIL": "BRA",
    "ECUADOR": "ECU",
    "COLOMBIA": "COL",
    "MEXICO": "MEX",
    "PERU": "PER",
    "VENEZUELA": "VEN",
    "VENEZUELA, RB": "VEN",
    "BOLIVARIAN REPUBLIC OF VENEZUELA": "VEN",
    "GHANA": "GHA",
    "KENYA": "KEN",
    "NIGERIA": "NGA",
    "SOUTH AFRICA": "ZAF",
    "ZAMBIA": "ZMB",
    "ETHIOPIA": "ETH",
    "MOZAMBIQUE": "MOZ",
    "INDIA": "IND",
    "INDONESIA": "IDN",
    "MALAYSIA": "MYS",
    "PAKISTAN": "PAK",
    "SRI LANKA": "LKA",
    "PHILIPPINES": "PHL",
    "EGYPT": "EGY",
    "EGYPT, ARAB REP.": "EGY",
    "EGYPT, ARAB REP": "EGY",
    "ARAB REPUBLIC OF EGYPT": "EGY",
    "LEBANON": "LBN",
    "TUNISIA": "TUN",
    "JORDAN": "JOR",
    "TURKEY": "TUR",
    "TURKIYE": "TUR",
    "REPUBLIC OF TURKEY": "TUR",
    "UKRAINE": "UKR",
    "RUSSIA": "RUS",
    "RUSSIAN FEDERATION": "RUS",
    "POLAND": "POL",
    "HUNGARY": "HUN",
    "ROMANIA": "ROU",
    "DOMINICAN REPUBLIC": "DOM",
    "JAMAICA": "JAM",
    # useful non-target names that must be ignored or skipped
    "WORLD": None,
    "KOREA, REP.": None,
    "SLOVAK REPUBLIC": None,
    "SUB-SAHARAN AFRICA": None,
}

KNOWN_CRISIS_MONTHS = {
    ("ARG", 2001): "2001-11",
    ("ARG", 2014): "2014-07",
    ("ARG", 2020): "2020-05",
    ("ECU", 2008): "2008-12",
    ("ECU", 2020): "2020-04",
    ("GHA", 2022): "2022-12",
    ("LKA", 2022): "2022-05",
    ("LBN", 2020): "2020-03",
    ("RUS", 1998): "1998-08",
    ("UKR", 2015): "2015-09",
    ("VEN", 2017): "2017-11",
    ("ZMB", 2020): "2020-11",
    ("MOZ", 2016): "2016-04",
    ("ETH", 2023): "2023-12",
    ("DOM", 2005): "2005-04",
    ("JAM", 2010): "2010-02",
    ("JAM", 2013): "2013-02",
    ("PAK", 1999): "1999-01",
    ("TUR", 2001): "2001-02",
    ("EGY", 2016): "2016-11",
    ("TUN", 2013): "2013-06",
    ("KEN", 2024): "2024-07",
}

TARGET_START = pd.Timestamp("2000-01-01")
TARGET_END = pd.Timestamp("2024-12-01")


# =============================================================================
# Utilities
# =============================================================================

def log(msg: str) -> None:
    print(f"[PIPELINE] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_country_name(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    value = re.sub(r"\s+", " ", value)
    return value


def map_country_to_iso3(value: str) -> Optional[str]:
    key = normalize_country_name(value)
    if key in COUNTRY_VARIANTS:
        return COUNTRY_VARIANTS[key]
    return None


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def identify_time_columns(columns: Iterable[str]) -> list[str]:
    time_cols = []
    for c in columns:
        c = str(c)
        if re.fullmatch(r"\d{4}", c):
            time_cols.append(c)
        elif re.fullmatch(r"\d{4}-Q[1-4]", c):
            time_cols.append(c)
        elif re.fullmatch(r"\d{4}-M\d{2}", c):
            time_cols.append(c)
    return time_cols


def parse_period_to_month_start(period: str) -> pd.Timestamp:
    period = str(period)
    if re.fullmatch(r"\d{4}-M\d{2}", period):
        return pd.Timestamp(period.replace("-M", "-") + "-01")
    if re.fullmatch(r"\d{4}-Q[1-4]", period):
        year = int(period[:4])
        q = int(period[-1])
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return pd.Timestamp(year=year, month=month, day=1)
    if re.fullmatch(r"\d{4}", period):
        return pd.Timestamp(year=int(period), month=1, day=1)
    return pd.NaT


def expand_annual_or_quarterly_to_monthly(df: pd.DataFrame,
                                          iso_col: str,
                                          date_col: str,
                                          value_cols: list[str]) -> pd.DataFrame:
    """
    Expand rows that are stamped at annual or quarterly period starts to
    monthly rows by forward filling within country over the target date range.
    """
    out = []
    for iso, g in df.groupby(iso_col):
        g = g.sort_values(date_col).copy()
        full = pd.DataFrame({iso_col: iso,
                             date_col: pd.date_range(g[date_col].min(),
                                                     g[date_col].max(),
                                                     freq="MS")})
        full = full.merge(g, on=[iso_col, date_col], how="left")
        full[value_cols] = full[value_cols].ffill()
        out.append(full)
    if not out:
        return pd.DataFrame(columns=[iso_col, date_col] + value_cols)
    return pd.concat(out, ignore_index=True)


def build_backbone() -> pd.DataFrame:
    months = pd.date_range(TARGET_START, TARGET_END, freq="MS")
    rows = []
    for iso3 in ISO3_LIST:
        for ym in months:
            rows.append({
                "iso3": iso3,
                "year_month": ym,
                "country_name": CANONICAL_BY_ISO3[iso3],
                "year": ym.year,
                "month_num": ym.month,
            })
    return pd.DataFrame(rows)


def missingness_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        s = df[col]
        non_null = df.loc[s.notna(), "year_month"] if "year_month" in df.columns else pd.Series(dtype="datetime64[ns]")
        records.append({
            "feature": col,
            "missing_pct": round(float(s.isna().mean() * 100), 6),
            "countries_with_data": int(df.loc[s.notna(), "iso3"].nunique()) if "iso3" in df.columns else np.nan,
            "non_null_count": int(s.notna().sum()),
            "min_date_with_data": non_null.min() if len(non_null) else pd.NaT,
            "max_date_with_data": non_null.max() if len(non_null) else pd.NaT,
        })
    return pd.DataFrame(records).sort_values(["missing_pct", "feature"], ascending=[False, True])


def percentile_within_month(df: pd.DataFrame, col: str, out_col: str) -> None:
    df[out_col] = df.groupby("year_month")[col].rank(pct=True)


def month_diff(future_date: pd.Timestamp, current_date: pd.Timestamp) -> int:
    return (future_date.year - current_date.year) * 12 + (future_date.month - current_date.month)


def save_npz_windows(df: pd.DataFrame,
                     split_name: str,
                     feature_cols: list[str],
                     out_dir: Path,
                     lookback: int = 24) -> None:
    df = df.sort_values(["iso3", "year_month"]).copy()
    X_list, y_list, meta_rows = [], [], []
    for iso, g in df.groupby("iso3"):
        g = g.reset_index(drop=True)
        values = g[feature_cols].to_numpy(dtype=float)
        labels = g["distress_within_12m"].to_numpy(dtype=int)
        dates = g["year_month"].astype(str).tolist()

        for i in range(lookback - 1, len(g)):
            window = values[i - lookback + 1:i + 1]
            if np.isnan(window).any():
                # Because missingness should already be handled, NaNs here are a sign
                # that the panel needs more attention. Skip conservatively.
                continue
            X_list.append(window)
            y_list.append(labels[i])
            meta_rows.append({"iso3": iso, "window_end_month": dates[i], "label": int(labels[i])})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.savez(out_dir / f"{split_name}_windows.npz", X=X, y=y)
    pd.DataFrame(meta_rows).to_csv(out_dir / f"{split_name}_meta.csv", index=False)
    log(f"Saved {split_name} windows: X shape={X.shape}, y shape={y.shape}")


# =============================================================================
# Archive extraction
# =============================================================================

def extract_archives(raw_zip: Path, bop_zip: Optional[Path], extract_root: Path) -> tuple[Path, Optional[Path]]:
    ensure_dir(extract_root)

    raw_dir = extract_root / "raw_data"
    bop_dir = extract_root / "raw_bop"
    ensure_dir(raw_dir)

    log(f"Extracting {raw_zip.name}")
    with zipfile.ZipFile(raw_zip, "r") as zf:
        zf.extractall(raw_dir)

    if bop_zip is not None and bop_zip.exists():
        ensure_dir(bop_dir)
        log(f"Extracting {bop_zip.name}")
        with zipfile.ZipFile(bop_zip, "r") as zf:
            zf.extractall(bop_dir)
        return raw_dir, bop_dir
    return raw_dir, None


# =============================================================================
# File 1 - ACLED raw events
# =============================================================================

def process_acled(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    log("Processing ACLED raw event data")
    usecols = [
        "event_date", "country", "iso", "event_type", "sub_event_type",
        "disorder_type", "fatalities", "civilian_targeting"
    ]
    df = pd.read_csv(raw_file, usecols=usecols, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", dayfirst=False)
    df["year_month"] = df["event_date"].dt.to_period("M").dt.to_timestamp()

    # Numeric ACLED ISO is not ISO3, so use country-name mapping.
    df["iso3"] = df["country"].map(map_country_to_iso3)
    df = df[df["iso3"].isin(ISO3_LIST)].copy()

    df["fatalities"] = safe_to_numeric(df["fatalities"]).fillna(0)
    df["civilian_targeting_flag"] = df["civilian_targeting"].notna() & (df["civilian_targeting"].astype(str).str.strip() != "")

    grouped = df.groupby(["iso3", "year_month"], dropna=False)
    out = grouped.size().rename("acled_total_events").reset_index()

    def count_where(g: pd.DataFrame, event_type: str) -> int:
        return int((g["event_type"] == event_type).sum())

    extra = grouped.apply(lambda g: pd.Series({
        "acled_battles": count_where(g, "Battles"),
        "acled_explosions": count_where(g, "Explosions/Remote violence"),
        "acled_violence_against_civilians": count_where(g, "Violence against civilians"),
        "acled_protests": count_where(g, "Protests"),
        "acled_riots": count_where(g, "Riots"),
        "acled_strategic_developments": count_where(g, "Strategic developments"),
        "acled_total_fatalities": float(g["fatalities"].sum()),
        "acled_civilian_targeting_events": int(g["civilian_targeting_flag"].sum()),
    })).reset_index()

    out = out.merge(extra, on=["iso3", "year_month"], how="left")
    out["acled_fatalities_per_event"] = np.where(out["acled_total_events"] > 0,
                                                 out["acled_total_fatalities"] / out["acled_total_events"],
                                                 0.0)
    violence_numer = (
        out["acled_battles"].fillna(0)
        + out["acled_explosions"].fillna(0)
        + out["acled_violence_against_civilians"].fillna(0)
    )
    protest_numer = out["acled_protests"].fillna(0) + out["acled_riots"].fillna(0)
    out["acled_violence_share"] = np.where(out["acled_total_events"] > 0,
                                           violence_numer / out["acled_total_events"], 0.0)
    out["acled_protest_share"] = np.where(out["acled_total_events"] > 0,
                                          protest_numer / out["acled_total_events"], 0.0)

    out = out.sort_values(["iso3", "year_month"])
    out.to_csv(interim_dir / "acled_country_month.csv", index=False)
    log(f"Saved ACLED country-month file with shape {out.shape}")
    return out


# =============================================================================
# File 2 - Pre-aggregated political violence
# =============================================================================

def process_political_violence(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    log("Processing pre-aggregated political violence file")
    df = pd.read_excel(raw_file)
    month_map = {
        "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
        "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    }
    df["iso3"] = df["COUNTRY"].map(map_country_to_iso3)
    df["month_num"] = df["MONTH"].astype(str).str.upper().map(month_map)
    df["year_month"] = pd.to_datetime(
        dict(year=safe_to_numeric(df["YEAR"]), month=safe_to_numeric(df["month_num"]), day=1),
        errors="coerce"
    )
    df = df[df["iso3"].isin(ISO3_LIST)].copy()
    df = df.rename(columns={"EVENTS": "political_violence_events_preagg"})
    out = df[["iso3", "year_month", "political_violence_events_preagg"]].copy()
    out["political_violence_events_preagg"] = safe_to_numeric(out["political_violence_events_preagg"])
    out = out.dropna(subset=["year_month"]).sort_values(["iso3", "year_month"])
    out.to_csv(interim_dir / "political_violence_monthly.csv", index=False)
    log(f"Saved political violence monthly file with shape {out.shape}")
    return out


# =============================================================================
# IMF wide-file helpers
# =============================================================================

def read_imf_wide_csv(file_path: Path) -> pd.DataFrame:
    log(f"Loading IMF wide file: {file_path.name}")
    return pd.read_csv(file_path, low_memory=False)


def melt_imf(df: pd.DataFrame,
             country_col: str = "COUNTRY",
             additional_keep_cols: Optional[list[str]] = None) -> tuple[pd.DataFrame, list[str]]:
    additional_keep_cols = additional_keep_cols or []
    time_cols = identify_time_columns(df.columns)
    keep_cols = [country_col] + [c for c in additional_keep_cols if c in df.columns]
    melted = df[keep_cols + time_cols].melt(id_vars=keep_cols, var_name="period", value_name="value")
    melted["year_month"] = melted["period"].map(parse_period_to_month_start)
    melted["value"] = safe_to_numeric(melted["value"])
    melted = melted.dropna(subset=["year_month"])
    melted["iso3"] = melted[country_col].map(map_country_to_iso3)
    melted = melted[melted["iso3"].isin(ISO3_LIST)].copy()
    return melted, time_cols


def choose_best_series_per_country(df: pd.DataFrame,
                                   group_cols: list[str],
                                   score_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Choose the most complete series per country among candidate series.
    score_cols can include simple preference columns such as frequency_rank.
    """
    score_cols = score_cols or []
    completeness = df.groupby(group_cols)["value"].apply(lambda s: s.notna().sum()).rename("non_null_count").reset_index()
    tmp = df.merge(completeness, on=group_cols, how="left")
    sort_cols = ["iso3", "non_null_count"] + score_cols
    ascending = [True, False] + [False] * len(score_cols)
    tmp = tmp.sort_values(sort_cols, ascending=ascending)
    chosen = tmp.groupby("iso3", as_index=False).head(1)[group_cols].drop_duplicates()
    return df.merge(chosen, on=group_cols, how="inner")


def frequency_rank(series: pd.Series) -> pd.Series:
    mapping = {"M": 3, "Q": 2, "A": 1, "Y": 1}
    return series.astype(str).str[0].map(mapping).fillna(0)


# =============================================================================
# File 3 - CPI
# =============================================================================

def process_cpi(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    df = read_imf_wide_csv(raw_file)

    # Use descriptive metadata if available.
    desc = (
        df.get("FULL_DESCRIPTION", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("INDICATOR", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("SERIES_NAME", pd.Series("", index=df.index)).astype(str)
    ).str.lower()

    keep = desc.str.contains("consumer price|cpi|all items", regex=True)
    exclude = desc.str.contains("food|housing|transport|furnish|health|education|restaurants|clothing|recreation", regex=True)
    df = df[keep & ~exclude].copy()

    if "FREQUENCY" in df.columns:
        df["freq_rank"] = frequency_rank(df["FREQUENCY"])
    else:
        df["freq_rank"] = 0

    melted, _ = melt_imf(df, additional_keep_cols=["FREQUENCY", "SERIES_CODE", "INDICATOR", "FULL_DESCRIPTION", "freq_rank"])
    melted = melted.dropna(subset=["value"])
    best = choose_best_series_per_country(
        melted,
        group_cols=["iso3", "SERIES_CODE", "FREQUENCY", "INDICATOR", "FULL_DESCRIPTION", "freq_rank"],
        score_cols=["freq_rank"]
    )

    out = best[["iso3", "year_month", "value"]].rename(columns={"value": "cpi_index"}).copy()
    out = expand_annual_or_quarterly_to_monthly(out, "iso3", "year_month", ["cpi_index"])
    out["cpi_mom_pct"] = out.groupby("iso3")["cpi_index"].pct_change() * 100
    out["cpi_yoy_pct"] = out.groupby("iso3")["cpi_index"].pct_change(12) * 100
    out["cpi_yoy_acceleration"] = out.groupby("iso3")["cpi_yoy_pct"].diff(3)

    out = out[(out["year_month"] >= TARGET_START) & (out["year_month"] <= TARGET_END)]
    out.to_csv(interim_dir / "cpi_monthly.csv", index=False)
    log(f"Saved CPI monthly file with shape {out.shape}")
    return out


# =============================================================================
# File 4 - Exchange rates
# =============================================================================

def process_fx(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    df = read_imf_wide_csv(raw_file)
    desc = (
        df.get("FULL_DESCRIPTION", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("INDICATOR", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("SERIES_NAME", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("EXRATE", pd.Series("", index=df.index)).astype(str)
    ).str.lower()

    keep = desc.str.contains("exchange rate|official rate|market rate|domestic currency|national currency|usd", regex=True)
    prefer = (
        desc.str.contains("usd", regex=False).astype(int)
        + desc.str.contains("end-of-period|end of period", regex=True).astype(int)
        + desc.str.contains("official|market", regex=True).astype(int)
    )

    if "FREQUENCY" in df.columns:
        df["freq_rank"] = frequency_rank(df["FREQUENCY"])
    else:
        df["freq_rank"] = 0
    df["prefer_rank"] = prefer
    df = df[keep].copy()

    melted, _ = melt_imf(df, additional_keep_cols=["FREQUENCY", "SERIES_CODE", "INDICATOR", "FULL_DESCRIPTION", "prefer_rank", "freq_rank"])
    melted = melted.dropna(subset=["value"])
    best = choose_best_series_per_country(
        melted,
        group_cols=["iso3", "SERIES_CODE", "FREQUENCY", "INDICATOR", "FULL_DESCRIPTION", "prefer_rank", "freq_rank"],
        score_cols=["freq_rank", "prefer_rank"]
    )

    out = best[["iso3", "year_month", "value"]].rename(columns={"value": "fx_rate"}).copy()
    out = expand_annual_or_quarterly_to_monthly(out, "iso3", "year_month", ["fx_rate"])
    out["fx_mom_pct_change"] = out.groupby("iso3")["fx_rate"].pct_change() * 100
    out["fx_yoy_pct_change"] = out.groupby("iso3")["fx_rate"].pct_change(12) * 100
    out["fx_3m_pct_change"] = out.groupby("iso3")["fx_rate"].pct_change(3) * 100
    out["fx_6m_pct_change"] = out.groupby("iso3")["fx_rate"].pct_change(6) * 100
    out["fx_volatility_3m"] = out.groupby("iso3")["fx_mom_pct_change"].rolling(3).std().reset_index(level=0, drop=True)

    out = out[(out["year_month"] >= TARGET_START) & (out["year_month"] <= TARGET_END)]
    out.to_csv(interim_dir / "fx_monthly.csv", index=False)
    log(f"Saved FX monthly file with shape {out.shape}")
    return out


# =============================================================================
# File 5 - WEO
# =============================================================================

def process_weo(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    df = read_imf_wide_csv(raw_file)
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    label_cols = [c for c in ["INDICATOR", "FULL_DESCRIPTION", "SERIES_NAME", "SERIES_CODE"] if c in df.columns]

    desc = pd.Series("", index=df.index)
    for c in label_cols:
        desc = desc + " " + df[c].astype(str)
    desc = desc.str.lower()

    indicator_map = {
        "weo_gdp_growth": r"gross domestic product, constant prices.*percent change|real gdp growth|gdp.*constant prices.*percent change",
        "weo_inflation": r"inflation, average consumer prices",
        "weo_govt_debt_gdp": r"general government gross debt.*percent of gdp",
        "weo_fiscal_balance_gdp": r"general government net lending/borrowing.*percent of gdp",
        "weo_current_account_gdp": r"current account balance.*percent of gdp",
        "weo_unemployment": r"unemployment rate",
        "weo_gdp_per_capita_ppp": r"gdp per capita, current international dollars|gdp per capita, ppp",
    }

    frames = []
    for new_name, pattern in indicator_map.items():
        sub = df[desc.str.contains(pattern, regex=True, na=False)].copy()
        if sub.empty:
            continue

        # Pick the single best row per country based on most non-null year observations.
        sub["non_null_count"] = sub[year_cols].notna().sum(axis=1)
        sub["iso3"] = sub["COUNTRY"].map(map_country_to_iso3)
        sub = sub[sub["iso3"].isin(ISO3_LIST)].copy()
        sub = sub.sort_values(["iso3", "non_null_count"], ascending=[True, False]).groupby("iso3", as_index=False).head(1)

        long_df = sub[["iso3"] + year_cols].melt(id_vars=["iso3"], var_name="year", value_name=new_name)
        long_df["year"] = safe_to_numeric(long_df["year"]).astype("Int64")
        long_df[new_name] = safe_to_numeric(long_df[new_name])
        frames.append(long_df)

    if not frames:
        out = pd.DataFrame(columns=["iso3", "year_month"])
        out.to_csv(interim_dir / "weo_monthly.csv", index=False)
        return out

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["iso3", "year"], how="outer")

    merged = merged.dropna(subset=["year"])
    merged["year"] = merged["year"].astype(int)
    merged = merged[merged["year"] <= 2024].copy()  # conservative anti-forecast rule

    rows = []
    value_cols = [c for c in merged.columns if c not in ["iso3", "year"]]
    for _, row in merged.iterrows():
        for month in range(1, 13):
            rows.append({"iso3": row["iso3"], "year_month": pd.Timestamp(year=int(row["year"]), month=month, day=1),
                         **{c: row[c] for c in value_cols},
                         "weo_is_annual": True})
    out = pd.DataFrame(rows)
    out = out[(out["year_month"] >= TARGET_START) & (out["year_month"] <= TARGET_END)]
    cols = ["iso3", "year_month", "weo_is_annual"] + sorted([c for c in out.columns if c not in ["iso3", "year_month", "weo_is_annual"]])
    out = out[cols]
    out.to_csv(interim_dir / "weo_monthly.csv", index=False)
    log(f"Saved WEO monthly file with shape {out.shape}")
    return out


# =============================================================================
# File 6 - External Debt Statistics
# =============================================================================

def process_external_debt(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    log("Processing World Bank External Debt Statistics")
    df = pd.read_excel(raw_file, sheet_name="Data")

    # Keep target countries only; World Bank uses ISO3 in Country Code.
    df = df[df["Country Code"].isin(ISO3_LIST)].copy()

    counterpart = df["Counterpart-Area Name"].astype(str).str.lower()
    df = df[counterpart.isin(["world", "all counterpart areas", "nan"]) | counterpart.str.contains("world", na=False)].copy()

    series_map = {
        "DT.DOD.DECT.CD": "edt_total_external_debt",
        "DT.DOD.DLXF.CD": "edt_long_term_debt",
        "DT.DOD.DSTC.CD": "edt_short_term_debt",
        "DT.TDS.DECT.CD": "edt_debt_service",
        "DT.DOD.DPPG.CD": "edt_ppg_debt",
        "DT.INT.DECT.CD": "edt_interest_payments",
        "DT.DOD.DECT.GN.ZS": "edt_debt_to_gni",
        "DT.DOD.DSTC.IR.ZS": "edt_short_term_to_reserves",
        "DT.TDS.DECT.EX.ZS": "edt_debt_service_to_exports",
    }
    name_fallbacks = {
        "edt_multilateral_debt": r"multilateral.*debt",
        "edt_reserves_to_debt": r"reserves.*external debt",
    }

    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    exact = df[df["Series Code"].isin(series_map.keys())].copy()
    exact["feature"] = exact["Series Code"].map(series_map)

    fallback_frames = []
    desc = df["Series Name"].astype(str).str.lower()
    for feat, patt in name_fallbacks.items():
        sub = df[desc.str.contains(patt, regex=True, na=False)].copy()
        if sub.empty:
            continue
        sub["feature"] = feat
        fallback_frames.append(sub)

    if fallback_frames:
        exact = pd.concat([exact] + fallback_frames, ignore_index=True)

    long_df = exact[["Country Code", "feature"] + year_cols].melt(
        id_vars=["Country Code", "feature"], var_name="year", value_name="value"
    )
    long_df["year"] = safe_to_numeric(long_df["year"]).astype("Int64")
    long_df["value"] = safe_to_numeric(long_df["value"])
    long_df = long_df.dropna(subset=["year"])
    long_df = long_df.rename(columns={"Country Code": "iso3"})
    wide = long_df.pivot_table(index=["iso3", "year"], columns="feature", values="value", aggfunc="first").reset_index()
    wide["year"] = wide["year"].astype(int)

    rows = []
    value_cols = [c for c in wide.columns if c not in ["iso3", "year"]]
    for _, row in wide.iterrows():
        for month in range(1, 13):
            rows.append({"iso3": row["iso3"], "year_month": pd.Timestamp(year=int(row["year"]), month=month, day=1),
                         **{c: row[c] for c in value_cols}})
    out = pd.DataFrame(rows).sort_values(["iso3", "year_month"])

    # Derived annual-change style features on the expanded monthly file.
    if "edt_total_external_debt" in out.columns:
        out["edt_total_debt_yoy_change"] = out.groupby("iso3")["edt_total_external_debt"].pct_change(12) * 100
    if {"edt_short_term_debt", "edt_total_external_debt"}.issubset(out.columns):
        out["edt_short_term_share"] = out["edt_short_term_debt"] / out["edt_total_external_debt"]
    if {"edt_ppg_debt", "edt_total_external_debt"}.issubset(out.columns):
        out["edt_ppg_share"] = out["edt_ppg_debt"] / out["edt_total_external_debt"]
    if "edt_debt_service_to_exports" in out.columns:
        out["edt_debt_service_ratio_change"] = out.groupby("iso3")["edt_debt_service_to_exports"].diff(12)

    out = out[(out["year_month"] >= TARGET_START) & (out["year_month"] <= TARGET_END)]
    out.to_csv(interim_dir / "external_debt_monthly.csv", index=False)
    log(f"Saved external debt monthly file with shape {out.shape}")
    return out


# =============================================================================
# File 7 - Interest rates
# =============================================================================

def process_interest_rates(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    df = read_imf_wide_csv(raw_file)
    desc = (
        df.get("FULL_DESCRIPTION", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("INDICATOR", pd.Series("", index=df.index)).astype(str)
        + " " + df.get("SERIES_NAME", pd.Series("", index=df.index)).astype(str)
    ).str.lower()

    rate_patterns = {
        "ir_policy_rate": r"discount rate|policy rate|central bank.*rate|monetary policy rate",
        "ir_lending_rate": r"lending rate",
        "ir_deposit_rate": r"deposit rate",
        "ir_tbill_rate": r"treasury bill|t-bill|tbill",
    }

    all_frames = []
    if "FREQUENCY" in df.columns:
        df["freq_rank"] = frequency_rank(df["FREQUENCY"])
    else:
        df["freq_rank"] = 0

    for out_name, pattern in rate_patterns.items():
        sub = df[desc.str.contains(pattern, regex=True, na=False)].copy()
        if sub.empty:
            continue
        melted, _ = melt_imf(sub, additional_keep_cols=["FREQUENCY", "SERIES_CODE", "INDICATOR", "FULL_DESCRIPTION", "freq_rank"])
        melted = melted.dropna(subset=["value"])
        best = choose_best_series_per_country(
            melted,
            group_cols=["iso3", "SERIES_CODE", "FREQUENCY", "INDICATOR", "FULL_DESCRIPTION", "freq_rank"],
            score_cols=["freq_rank"]
        )
        out = best[["iso3", "year_month", "value"]].rename(columns={"value": out_name})
        out = expand_annual_or_quarterly_to_monthly(out, "iso3", "year_month", [out_name])
        all_frames.append(out)

    if not all_frames:
        out = pd.DataFrame(columns=["iso3", "year_month"])
        out.to_csv(interim_dir / "interest_rates_monthly.csv", index=False)
        return out

    merged = all_frames[0]
    for f in all_frames[1:]:
        merged = merged.merge(f, on=["iso3", "year_month"], how="outer")

    if {"ir_lending_rate", "ir_deposit_rate"}.issubset(merged.columns):
        merged["ir_spread"] = merged["ir_lending_rate"] - merged["ir_deposit_rate"]

    merged = merged[(merged["year_month"] >= TARGET_START) & (merged["year_month"] <= TARGET_END)]
    merged.to_csv(interim_dir / "interest_rates_monthly.csv", index=False)
    log(f"Saved interest rates monthly file with shape {merged.shape}")
    return merged


# =============================================================================
# File 8 - BIS debt securities (optional / phase 2)
# =============================================================================

def process_bis(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    log("Processing BIS debt securities (optional)")
    try:
        df = pd.read_csv(raw_file, low_memory=False)
        area_col = "REF_AREA" if "REF_AREA" in df.columns else None
        time_col = None
        for candidate in ["TIME_PERIOD:Period", "TIME_PERIOD", "TIME_PERIODS", "TIME"]:
            if candidate in df.columns:
                time_col = candidate
                break
        value_col = None
        for candidate in ["OBS_VALUE", "Value", "OBS_VALUE:Observation Value", "OBS_VALUE (FREQ)"]:
            if candidate in df.columns:
                value_col = candidate
                break

        if not all([area_col, time_col, value_col]):
            raise ValueError("Could not identify key BIS columns")

        # ISO2 to ISO3 subset used for these target countries.
        iso2_to_iso3 = {
            "AR": "ARG", "BR": "BRA", "EC": "ECU", "CO": "COL", "MX": "MEX", "PE": "PER", "VE": "VEN",
            "GH": "GHA", "KE": "KEN", "NG": "NGA", "ZA": "ZAF", "ZM": "ZMB", "ET": "ETH", "MZ": "MOZ",
            "IN": "IND", "ID": "IDN", "MY": "MYS", "PK": "PAK", "LK": "LKA", "PH": "PHL",
            "EG": "EGY", "LB": "LBN", "TN": "TUN", "JO": "JOR", "TR": "TUR", "UA": "UKR", "RU": "RUS",
            "PL": "POL", "HU": "HUN", "RO": "ROU", "DO": "DOM", "JM": "JAM",
        }

        df["iso3"] = df[area_col].astype(str).str.upper().map(iso2_to_iso3)
        df = df[df["iso3"].isin(ISO3_LIST)].copy()

        desc = df.astype(str).agg(" ".join, axis=1).str.lower()
        keep = desc.str.contains("general government|central government", regex=True) & desc.str.contains("debt securities|outstanding", regex=True)
        df = df[keep].copy()

        # Parse time periods like 2019-Q4, 2020-03, etc.
        def parse_bis_period(x: str) -> pd.Timestamp:
            x = str(x)
            if re.fullmatch(r"\d{4}-Q[1-4]", x):
                return parse_period_to_month_start(x)
            if re.fullmatch(r"\d{4}-\d{2}", x):
                return pd.Timestamp(x + "-01")
            if re.fullmatch(r"\d{4}", x):
                return pd.Timestamp(x + "-01-01")
            return pd.NaT

        df["year_month"] = df[time_col].map(parse_bis_period)
        df["bis_govt_debt_securities_outstanding"] = safe_to_numeric(df[value_col])
        out = df[["iso3", "year_month", "bis_govt_debt_securities_outstanding"]].dropna(subset=["year_month"])
        out = out.groupby(["iso3", "year_month"], as_index=False)["bis_govt_debt_securities_outstanding"].mean()
        out = expand_annual_or_quarterly_to_monthly(out, "iso3", "year_month", ["bis_govt_debt_securities_outstanding"])
        out = out[(out["year_month"] >= TARGET_START) & (out["year_month"] <= TARGET_END)]
        out.to_csv(interim_dir / "bis_debt_securities_monthly.csv", index=False)
        log(f"Saved BIS debt securities monthly file with shape {out.shape}")
        return out
    except Exception as e:
        log(f"Skipping BIS file because parsing failed: {e}")
        out = pd.DataFrame(columns=["iso3", "year_month", "bis_govt_debt_securities_outstanding"])
        out.to_csv(interim_dir / "bis_debt_securities_monthly.csv", index=False)
        return out


# =============================================================================
# Files 9-13 - BoC-BoE default database to labels
# =============================================================================

def parse_boc_boe_observations(file_path: Path) -> pd.DataFrame:
    """
    Bank of Canada / Bank of England files are not standard CSV files from row 1.
    They contain metadata, then a SERIES dictionary, then an OBSERVATIONS block.
    This parser finds the OBSERVATIONS line and reads the block that follows.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    obs_idx = None
    for i, line in enumerate(text):
        if line.strip().strip('"') == "OBSERVATIONS":
            obs_idx = i
            break
    if obs_idx is None:
        raise ValueError(f"OBSERVATIONS block not found in {file_path.name}")

    obs_text = "\n".join(text[obs_idx + 1:])
    df = pd.read_csv(io.StringIO(obs_text))
    return df


def process_boc_boe_label_files(raw_dir: Path, interim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Processing BoC-BoE sovereign default files for label construction")
    pattern_files = sorted(list(raw_dir.glob("BoC-BoE_DEBT_20*.csv")))
    records = []
    for fp in pattern_files:
        try:
            df = parse_boc_boe_observations(fp)
            country_col = [c for c in df.columns if "DEBT_COUNTRY" == c or c.endswith("DEBT_COUNTRY")]
            year_col = [c for c in df.columns if "DEBT_YEAR" == c or c.endswith("DEBT_YEAR")]
            total_cols = [c for c in df.columns if re.fullmatch(r"DEBT_TOTAL_\d{4}", c)]
            if not country_col or not year_col or not total_cols:
                log(f"Skipping {fp.name}: key label columns not found")
                continue
            country_col = country_col[0]
            year_col = year_col[0]
            total_col = total_cols[0]

            tmp = df[[country_col, year_col, total_col]].copy()
            tmp = tmp.rename(columns={country_col: "country_name_raw", year_col: "year", total_col: "debt_in_default_usd"})
            tmp["iso3"] = tmp["country_name_raw"].map(map_country_to_iso3)
            tmp["year"] = safe_to_numeric(tmp["year"]).astype("Int64")
            tmp["debt_in_default_usd"] = safe_to_numeric(tmp["debt_in_default_usd"]).fillna(0)
            tmp = tmp[tmp["iso3"].isin(ISO3_LIST)].copy()
            tmp["is_in_default"] = (tmp["debt_in_default_usd"] > 0).astype(int)
            tmp["source"] = fp.name
            records.append(tmp[["iso3", "year", "debt_in_default_usd", "is_in_default", "source"]])
            log(f"Parsed label file {fp.name} with shape {tmp.shape}")
        except Exception as e:
            log(f"Failed to parse {fp.name}: {e}")

    if not records:
        raise RuntimeError("No BoC-BoE label files could be parsed")

    all_yearly = pd.concat(records, ignore_index=True)
    all_yearly = (all_yearly
                  .sort_values(["iso3", "year", "source"])
                  .drop_duplicates(subset=["iso3", "year"], keep="last")
                  .reset_index(drop=True))

    all_yearly["prev_is_in_default"] = all_yearly.groupby("iso3")["is_in_default"].shift(1).fillna(0)
    starts = all_yearly[(all_yearly["is_in_default"] == 1) & (all_yearly["prev_is_in_default"] == 0)].copy()

    event_rows = []
    for _, row in starts.iterrows():
        iso3 = row["iso3"]
        year = int(row["year"])
        if (iso3, year) in KNOWN_CRISIS_MONTHS:
            start_month = pd.Timestamp(KNOWN_CRISIS_MONTHS[(iso3, year)] + "-01")
            source = "KNOWN_CRISIS_MONTHS"
            event_type = "known_crisis_month"
        else:
            start_month = pd.Timestamp(year=year, month=1, day=1)
            source = row["source"]
            event_type = "default_or_distress_start"

        event_rows.append({
            "iso3": iso3,
            "start_month": start_month,
            "end_year": year,
            "source": source,
            "event_type": event_type,
        })

    distress_events = pd.DataFrame(event_rows).sort_values(["iso3", "start_month"]).reset_index(drop=True)
    distress_events.to_csv(interim_dir / "distress_events.csv", index=False)

    # Build complete monthly labels.
    backbone = build_backbone()[["iso3", "year_month"]].copy()
    labels = backbone.copy()
    labels["distress_within_12m"] = 0
    labels["in_distress_now"] = 0
    labels["months_to_next_distress"] = np.nan

    start_map = defaultdict(list)
    for _, row in distress_events.iterrows():
        start_map[row["iso3"]].append(pd.Timestamp(row["start_month"]))

    current_default_lookup = all_yearly[["iso3", "year", "is_in_default"]].copy()

    for iso in ISO3_LIST:
        event_months = sorted(start_map.get(iso, []))
        iso_idx = labels["iso3"] == iso
        iso_rows = labels.loc[iso_idx].copy()

        # Forward-looking label.
        out_y = []
        out_mtn = []
        for ym in iso_rows["year_month"]:
            future_events = [d for d in event_months if ym < d <= ym + pd.DateOffset(months=12)]
            if future_events:
                out_y.append(1)
                out_mtn.append(min(month_diff(d, ym) for d in future_events))
            else:
                out_y.append(0)
                out_mtn.append(np.nan)
        labels.loc[iso_idx, "distress_within_12m"] = out_y
        labels.loc[iso_idx, "months_to_next_distress"] = out_mtn

        # Current distress status using annual in-default flag.
        yearly_status = current_default_lookup[current_default_lookup["iso3"] == iso].set_index("year")["is_in_default"].to_dict()
        labels.loc[iso_idx, "in_distress_now"] = labels.loc[iso_idx, "year_month"].dt.year.map(yearly_status).fillna(0).astype(int)

    labels.to_csv(interim_dir / "labels_monthly.csv", index=False)
    log(f"Saved labels monthly file with shape {labels.shape}")
    return distress_events, labels


# =============================================================================
# Large raw data - IMF BOP
# =============================================================================

def process_bop(raw_file: Path, interim_dir: Path) -> pd.DataFrame:
    """
    BOP file is very large. This function uses chunked reading and row-level
    keyword selection so we do not have to materialize unnecessary rows in memory.
    """
    log("Processing large IMF Balance of Payments file")

    desired_series = {
        "bop_current_account_balance_usd": [r"current account balance"],
        "bop_reserve_assets_usd": [r"reserve assets"],
        "bop_fdi_liabilities_usd": [r"direct investment.*liabilities|fdi.*liabilities"],
        "bop_portfolio_liabilities_usd": [r"portfolio investment.*liabilities"],
        "bop_other_investment_loans_liabilities_usd": [r"other investment.*loan.*liabilities"],
        "bop_workers_remittances_credit_usd": [r"workers'? remittances.*credit|personal transfers.*credit"],
    }

    selection_rows = []
    chunks = pd.read_csv(raw_file, chunksize=2500, low_memory=False)

    for i, chunk in enumerate(chunks, start=1):
        if "COUNTRY" not in chunk.columns:
            raise ValueError("COUNTRY column missing from BOP file")
        chunk["iso3"] = chunk["COUNTRY"].map(map_country_to_iso3)
        chunk = chunk[chunk["iso3"].isin(ISO3_LIST)].copy()
        if chunk.empty:
            continue

        desc = (
            chunk.get("FULL_DESCRIPTION", pd.Series("", index=chunk.index)).astype(str)
            + " " + chunk.get("INDICATOR", pd.Series("", index=chunk.index)).astype(str)
            + " " + chunk.get("SERIES_NAME", pd.Series("", index=chunk.index)).astype(str)
            + " " + chunk.get("FUNCTIONAL_CAT", pd.Series("", index=chunk.index)).astype(str)
            + " " + chunk.get("ACCOUNTING_ENTRY", pd.Series("", index=chunk.index)).astype(str)
        ).str.lower()

        keep_idx = pd.Series(False, index=chunk.index)
        assigned_name = pd.Series("", index=chunk.index, dtype="object")
        for out_name, patterns in desired_series.items():
            mask = pd.Series(False, index=chunk.index)
            for patt in patterns:
                mask = mask | desc.str.contains(patt, regex=True, na=False)
            assigned_name.loc[mask] = out_name
            keep_idx = keep_idx | mask

        sub = chunk.loc[keep_idx].copy()
        if sub.empty:
            continue

        sub["selected_feature"] = assigned_name.loc[sub.index]
        sub["freq_rank"] = frequency_rank(sub["FREQUENCY"]) if "FREQUENCY" in sub.columns else 0
        time_cols = identify_time_columns(sub.columns)
        if not time_cols:
            continue

        # Choose best series per country+feature before melting the full chunk.
        sub["non_null_count"] = sub[time_cols].notna().sum(axis=1)
        best_keys = (sub.sort_values(["iso3", "selected_feature", "non_null_count", "freq_rank"],
                                     ascending=[True, True, False, False])
                       .groupby(["iso3", "selected_feature"], as_index=False)
                       .head(1))
        selection_rows.append(best_keys)

        if i % 50 == 0:
            log(f"BOP chunk scan progress: {i} chunks")

    if not selection_rows:
        log("No BOP rows matched selection criteria")
        out = pd.DataFrame(columns=["iso3", "year_month"])
        out.to_csv(interim_dir / "bop_monthly.csv", index=False)
        return out

    selected = pd.concat(selection_rows, ignore_index=True)
    selected = selected.sort_values(["iso3", "selected_feature", "non_null_count", "freq_rank"],
                                    ascending=[True, True, False, False])
    selected = selected.drop_duplicates(subset=["iso3", "selected_feature"], keep="first")

    # Save the selected series catalog for auditability.
    selection_cols = [c for c in ["iso3", "COUNTRY", "selected_feature", "SERIES_CODE", "INDICATOR", "FREQUENCY", "FULL_DESCRIPTION", "non_null_count"] if c in selected.columns]
    selected[selection_cols].to_csv(interim_dir / "bop_series_selection.csv", index=False)

    time_cols = identify_time_columns(selected.columns)
    long_df = selected[["iso3", "selected_feature"] + time_cols].melt(
        id_vars=["iso3", "selected_feature"], var_name="period", value_name="value"
    )
    long_df["year_month"] = long_df["period"].map(parse_period_to_month_start)
    long_df["value"] = safe_to_numeric(long_df["value"])
    long_df = long_df.dropna(subset=["year_month"])
    wide = long_df.pivot_table(index=["iso3", "year_month"], columns="selected_feature", values="value", aggfunc="first").reset_index()
    value_cols = [c for c in wide.columns if c not in ["iso3", "year_month"]]
    wide = expand_annual_or_quarterly_to_monthly(wide, "iso3", "year_month", value_cols)

    if {"bop_fdi_liabilities_usd", "bop_portfolio_liabilities_usd", "bop_other_investment_loans_liabilities_usd"}.issubset(wide.columns):
        wide["bop_total_liability_inflows_usd"] = (
            wide["bop_fdi_liabilities_usd"].fillna(0)
            + wide["bop_portfolio_liabilities_usd"].fillna(0)
            + wide["bop_other_investment_loans_liabilities_usd"].fillna(0)
        )
    if "bop_current_account_balance_usd" in wide.columns:
        wide["bop_current_account_balance_usd_3m_change"] = wide.groupby("iso3")["bop_current_account_balance_usd"].diff(3)
    if "bop_reserve_assets_usd" in wide.columns:
        wide["bop_reserve_assets_usd_3m_change"] = wide.groupby("iso3")["bop_reserve_assets_usd"].diff(3)
        wide["bop_reserve_assets_3m_pct_change"] = wide.groupby("iso3")["bop_reserve_assets_usd"].pct_change(3) * 100
    if "bop_total_liability_inflows_usd" in wide.columns:
        wide["bop_total_liability_inflows_usd_3m_change"] = wide.groupby("iso3")["bop_total_liability_inflows_usd"].diff(3)

    wide = wide[(wide["year_month"] >= TARGET_START) & (wide["year_month"] <= TARGET_END)]
    wide.to_csv(interim_dir / "bop_monthly.csv", index=False)

    notes = [
        "BOP cleaning notes",
        "==================",
        f"Input file: {raw_file.name}",
        "Selection strategy:",
        "- Large IMF BOP file scanned in chunks to avoid excessive memory use.",
        "- Candidate rows filtered to target countries first.",
        "- Rows matched to a small list of macro-external financing concepts.",
        "- One best series per country per feature chosen using completeness and frequency preference.",
        "- Quarterly and annual values expanded to monthly by within-country forward fill.",
    ]
    (interim_dir / "bop_cleaning_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    log(f"Saved BOP monthly file with shape {wide.shape}")
    return wide


# =============================================================================
# Merge, impute, engineer, split
# =============================================================================

def merge_all_sources(interim_dir: Path, processed_dir: Path) -> pd.DataFrame:
    log("Creating full country-month backbone")
    panel = build_backbone()

    # Ordered joins, matching the cleaning specification.
    join_order = [
        ("labels_monthly.csv", ["distress_within_12m", "in_distress_now", "months_to_next_distress"]),
        ("acled_country_month.csv", None),
        ("political_violence_monthly.csv", None),
        ("cpi_monthly.csv", None),
        ("fx_monthly.csv", None),
        ("weo_monthly.csv", None),
        ("external_debt_monthly.csv", None),
        ("interest_rates_monthly.csv", None),
        ("bis_debt_securities_monthly.csv", None),
        ("bop_monthly.csv", None),
    ]

    source_map = {}
    for file_name, _ in join_order:
        file_path = interim_dir / file_name
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path, parse_dates=["year_month"])
        cols_before = set(panel.columns)
        panel = panel.merge(df, on=["iso3", "year_month"], how="left")
        new_cols = [c for c in panel.columns if c not in cols_before]
        for c in new_cols:
            source_map[c] = file_name.replace(".csv", "")

    # Derived interaction that requires merged sources.
    if {"ir_policy_rate", "cpi_yoy_pct"}.issubset(panel.columns):
        panel["ir_real_rate"] = panel["ir_policy_rate"] - panel["cpi_yoy_pct"]
        source_map["ir_real_rate"] = "derived_merged"

    if {"bop_reserve_assets_usd", "edt_total_external_debt"}.issubset(panel.columns):
        panel["bop_reserves_to_external_debt_ratio"] = panel["bop_reserve_assets_usd"] / panel["edt_total_external_debt"]
        source_map["bop_reserves_to_external_debt_ratio"] = "derived_merged"

    if {"bop_current_account_balance_usd", "bop_total_liability_inflows_usd"}.issubset(panel.columns):
        panel["bop_financing_gap_usd"] = panel["bop_current_account_balance_usd"] - panel["bop_total_liability_inflows_usd"]
        source_map["bop_financing_gap_usd"] = "derived_merged"

    panel = panel.sort_values(["iso3", "year_month"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in ["iso3", "year_month", "country_name", "year", "month_num"]]
    pre_miss = missingness_summary(panel, feature_cols)
    pre_miss["source"] = pre_miss["feature"].map(source_map)
    pre_miss.to_csv(processed_dir / "missingness_report.csv", index=False)

    # Drop overly sparse raw features. Important: never drop labels.
    label_cols = ["distress_within_12m", "in_distress_now", "months_to_next_distress"]
    drop_cols = pre_miss.loc[(pre_miss["missing_pct"] > 50) & (~pre_miss["feature"].isin(label_cols)), "feature"].tolist()
    if drop_cols:
        log(f"Dropping {len(drop_cols)} feature columns with >50% missingness")
        panel = panel.drop(columns=drop_cols, errors="ignore")

    # Create missingness indicator columns before imputation for variables that were meaningfully sparse.
    feature_cols = [c for c in panel.columns if c not in ["iso3", "year_month", "country_name", "year", "month_num"]]
    indicator_candidates = pre_miss.loc[(pre_miss["missing_pct"] > 10) & (pre_miss["feature"].isin(feature_cols)), "feature"].tolist()
    for col in indicator_candidates:
        panel[f"{col}_missing"] = panel[col].isna().astype(int)

    # Country-wise forward fill then backward fill up to 6 months.
    non_id_cols = [c for c in panel.columns if c not in ["iso3", "year_month", "country_name", "year", "month_num"]]
    for col in non_id_cols:
        panel[col] = panel.groupby("iso3")[col].ffill()
        panel[col] = panel.groupby("iso3")[col].bfill(limit=6)

    # Cross-sectional monthly median fill for remaining numeric gaps.
    numeric_cols = [c for c in non_id_cols if pd.api.types.is_numeric_dtype(panel[c])]
    for col in numeric_cols:
        panel[col] = panel.groupby("year_month")[col].transform(lambda s: s.fillna(s.median()))
    # Remaining labels: months_to_next_distress should stay NaN when no future event exists.
    # Leave it as is; it is informational and not a model feature.

    # Additional rolling features per the cleaning specification.
    panel = panel.sort_values(["iso3", "year_month"]).reset_index(drop=True)

    if "acled_total_events" in panel.columns:
        panel["acled_events_3m_avg"] = panel.groupby("iso3")["acled_total_events"].rolling(3).mean().reset_index(level=0, drop=True)
        panel["acled_events_6m_avg"] = panel.groupby("iso3")["acled_total_events"].rolling(6).mean().reset_index(level=0, drop=True)
        panel["acled_events_3m_change"] = panel.groupby("iso3")["acled_events_3m_avg"].pct_change(3) * 100
    if "acled_total_fatalities" in panel.columns:
        panel["acled_fatalities_3m_avg"] = panel.groupby("iso3")["acled_total_fatalities"].rolling(3).mean().reset_index(level=0, drop=True)
    if "acled_protests" in panel.columns:
        panel["acled_protest_trend_3m"] = panel.groupby("iso3")["acled_protests"].diff(3)

    if "edt_total_external_debt" in panel.columns:
        panel["edt_total_debt_6m_pct_change"] = panel.groupby("iso3")["edt_total_external_debt"].pct_change(6) * 100
    if "edt_debt_service" in panel.columns:
        panel["edt_debt_service_6m_pct_change"] = panel.groupby("iso3")["edt_debt_service"].pct_change(6) * 100
    if "edt_short_term_to_reserves" in panel.columns:
        panel["edt_short_term_ratio_change_6m"] = panel.groupby("iso3")["edt_short_term_to_reserves"].diff(6)

    if "fx_rate" in panel.columns:
        panel["fx_3m_depreciation"] = panel.groupby("iso3")["fx_rate"].pct_change(3) * 100
        panel["fx_6m_depreciation"] = panel.groupby("iso3")["fx_rate"].pct_change(6) * 100
    if "cpi_yoy_pct" in panel.columns:
        panel["cpi_inflation_acceleration"] = panel.groupby("iso3")["cpi_yoy_pct"].diff(3)

    # Percentile-based stress index
    if "edt_debt_to_gni" in panel.columns:
        percentile_within_month(panel, "edt_debt_to_gni", "edt_debt_to_gni_pctile")
    if "edt_short_term_to_reserves" in panel.columns:
        percentile_within_month(panel, "edt_short_term_to_reserves", "edt_short_term_to_reserves_pctile")
    if "fx_3m_depreciation" in panel.columns:
        percentile_within_month(panel, "fx_3m_depreciation", "fx_3m_depreciation_pctile")

    stress_parts = [c for c in ["edt_debt_to_gni_pctile", "edt_short_term_to_reserves_pctile", "fx_3m_depreciation_pctile"] if c in panel.columns]
    if stress_parts:
        panel["debt_stress_index"] = panel[stress_parts].sum(axis=1)

    if {"acled_total_events", "edt_debt_to_gni"}.issubset(panel.columns):
        panel["instability_x_debt"] = panel["acled_total_events"] * panel["edt_debt_to_gni"]

    # Drop zero-variance columns after engineering, excluding labels and IDs.
    non_id_non_label = [c for c in panel.columns if c not in ["iso3", "year_month", "country_name", "year", "month_num", "distress_within_12m", "in_distress_now", "months_to_next_distress"]]
    zero_var = [c for c in non_id_non_label if panel[c].nunique(dropna=False) <= 1]
    if zero_var:
        log(f"Dropping zero-variance columns: {zero_var}")
        panel = panel.drop(columns=zero_var)

    # Save summary statistics.
    numeric_cols = [c for c in panel.columns if pd.api.types.is_numeric_dtype(panel[c])]
    summary = pd.DataFrame({
        "feature": numeric_cols,
        "mean": [panel[c].mean(skipna=True) for c in numeric_cols],
        "std": [panel[c].std(skipna=True) for c in numeric_cols],
        "min": [panel[c].min(skipna=True) for c in numeric_cols],
        "max": [panel[c].max(skipna=True) for c in numeric_cols],
        "missing_pct": [panel[c].isna().mean() * 100 for c in numeric_cols],
    })
    summary.to_csv(processed_dir / "summary_statistics.csv", index=False)

    # Modality map
    modality_map = {
        "macro": [],
        "debt": [],
        "political": [],
        "engineered": [],
        "labels": ["distress_within_12m", "in_distress_now", "months_to_next_distress"],
    }
    for c in panel.columns:
        if c in modality_map["labels"] or c in ["iso3", "year_month", "country_name", "year", "month_num"]:
            continue
        lc = c.lower()
        if lc.startswith(("cpi_", "fx_", "weo_", "ir_", "bop_")):
            modality_map["macro"].append(c)
        elif lc.startswith(("edt_", "bis_")):
            modality_map["debt"].append(c)
        elif lc.startswith(("acled_", "political_violence_")):
            modality_map["political"].append(c)
        else:
            modality_map["engineered"].append(c)

    (processed_dir / "feature_modality_map.json").write_text(json.dumps(modality_map, indent=2), encoding="utf-8")
    feature_names = [c for c in panel.columns if c not in ["iso3", "year_month", "country_name", "year", "month_num"]]
    (processed_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    panel.to_csv(processed_dir / "panel_full.csv", index=False)
    try:
        panel.to_parquet(processed_dir / "panel_full.parquet", index=False)
    except Exception:
        (processed_dir / "panel_full.parquet.unavailable.txt").write_text(
            "Parquet export was skipped because no parquet engine was available in the runtime environment.",
            encoding="utf-8"
        )

    return panel


def split_and_save(panel: pd.DataFrame, processed_dir: Path) -> None:
    log("Creating time-based train / validation / test splits")

    train = panel[(panel["year_month"] >= pd.Timestamp("2000-01-01")) & (panel["year_month"] <= pd.Timestamp("2016-12-01"))].copy()
    val = panel[(panel["year_month"] >= pd.Timestamp("2017-01-01")) & (panel["year_month"] <= pd.Timestamp("2019-12-01"))].copy()
    test = panel[(panel["year_month"] >= pd.Timestamp("2020-01-01")) & (panel["year_month"] <= pd.Timestamp("2024-12-01"))].copy()

    train.to_csv(processed_dir / "train_flat.csv", index=False)
    val.to_csv(processed_dir / "val_flat.csv", index=False)
    test.to_csv(processed_dir / "test_flat.csv", index=False)

    class_dist = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        class_dist[name] = {
            "rows": int(len(df)),
            "y1": int(df["distress_within_12m"].sum()),
            "y0": int((df["distress_within_12m"] == 0).sum()),
            "positive_countries": sorted(df.loc[df["distress_within_12m"] == 1, "iso3"].unique().tolist()),
        }
        log(f"{name}: rows={class_dist[name]['rows']} y1={class_dist[name]['y1']} y0={class_dist[name]['y0']}")
    (processed_dir / "class_distribution.json").write_text(json.dumps(class_dist, indent=2), encoding="utf-8")

    # Exclude informational label fields from model features.
    feature_cols = [
        c for c in panel.columns
        if c not in ["iso3", "year_month", "country_name", "year", "month_num",
                     "distress_within_12m", "in_distress_now", "months_to_next_distress"]
    ]
    save_npz_windows(train, "train", feature_cols, processed_dir, lookback=24)
    save_npz_windows(val, "val", feature_cols, processed_dir, lookback=24)
    save_npz_windows(test, "test", feature_cols, processed_dir, lookback=24)


# =============================================================================
# Main
# =============================================================================

def locate_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename} under {root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and arrange sovereign debt distress datasets")
    parser.add_argument("--raw-zip", required=True, help="Path to Raw Data zip")
    parser.add_argument("--bop-zip", required=False, help="Path to Large Raw Data zip (BOP)")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    args = parser.parse_args()

    raw_zip = Path(args.raw_zip)
    bop_zip = Path(args.bop_zip) if args.bop_zip else None
    output_root = Path(args.output_dir)

    interim_dir = ensure_dir(output_root / "data" / "interim")
    processed_dir = ensure_dir(output_root / "data" / "processed")
    extract_root = ensure_dir(output_root / "_extracted")

    raw_dir, bop_dir = extract_archives(raw_zip, bop_zip, extract_root)

    raw_scope = raw_dir / "Raw Data" if (raw_dir / "Raw Data").exists() else raw_dir
    bop_scope = bop_dir / "Large Raw Data" if (bop_dir is not None and (bop_dir / "Large Raw Data").exists()) else bop_dir

    # Process each source independently.
    process_acled(locate_file(raw_scope, "ACLED Data_2026-04-03 (1).csv"), interim_dir)
    process_political_violence(locate_file(raw_scope, "number_of_political_violence_events_by_country-month-year_as-of-27Mar2026.xlsx"), interim_dir)
    process_cpi(locate_file(raw_scope, "Consumer_Price_Index.csv"), interim_dir)
    process_fx(locate_file(raw_scope, "Exchange_rate.csv"), interim_dir)
    process_weo(locate_file(raw_scope, "WEO.csv"), interim_dir)
    process_external_debt(locate_file(raw_scope, "External Debt Statistics_ALL COUNTRIES.xlsx"), interim_dir)
    process_interest_rates(locate_file(raw_scope, "dataset_2026-04-03T02_23_22.848329059Z_DEFAULT_INTEGRATION_IMF.STA_MFS_IR_8.0.1.csv"), interim_dir)
    process_bis(locate_file(raw_scope, "Debt_Security_Statistics.csv"), interim_dir)
    process_boc_boe_label_files(raw_scope, interim_dir)
    if bop_scope is not None:
        process_bop(locate_file(bop_scope, "Balance_of_Payments(BOP).csv"), interim_dir)

    # Merge and save.
    panel = merge_all_sources(interim_dir, processed_dir)
    split_and_save(panel, processed_dir)

    # Final validation prints
    log(f"Final panel shape: {panel.shape}")
    log(f"Unique countries in final panel: {panel['iso3'].nunique()} / expected {len(ISO3_LIST)}")
    log(f"Date range: {panel['year_month'].min()} to {panel['year_month'].max()}")

    feature_cols = [c for c in panel.columns if c not in ["iso3", "year_month", "country_name", "year", "month_num",
                                                          "distress_within_12m", "in_distress_now", "months_to_next_distress"]]
    log(f"Feature count (excluding identifiers and labels): {len(feature_cols)}")

    zero_event_countries = panel.groupby("iso3")["distress_within_12m"].sum()
    log(f"Countries with zero positive labels: {sorted(zero_event_countries[zero_event_countries == 0].index.tolist())}")

    # Correlations for audit.
    numeric_cols = [c for c in panel.columns if pd.api.types.is_numeric_dtype(panel[c])]
    if "distress_within_12m" in panel.columns and len(numeric_cols) > 1:
        corr = panel[numeric_cols].corr(numeric_only=True)["distress_within_12m"].drop("distress_within_12m", errors="ignore")
        corr = corr.abs().sort_values(ascending=False)
        log("Top 10 absolute correlations with target:")
        print(corr.head(10))

    log("Pipeline complete.")


if __name__ == "__main__":
    main()
