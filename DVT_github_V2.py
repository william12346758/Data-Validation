"""
Created on Tue Jul 15 11:40:19 2025

@author: LWu
The script is written by Le Wu with assistance of GPT-O3

To run this file:
    Requirements: Python ≥3.9 · pandas ≥2.2 · streamlit ≥1.35
    1. pip install all packages below
    2. Put the script in your Python work directory or desired folder
    3. Type the following line in (conda) command prompt: 
        streamlit run DVT_V2.py
    
    
Main features of this tool:
    • Indicator header/synonym detection
    • Automatic wide‑to‑long converter for year columns
    • Automatic numeric coercion for indicator columns if in a wide format
    • Gap, outlier, and structural‑break scans per decomposed series group
    • Custom rules on numerical relations can be applied to examine the data
"""

from __future__ import annotations

import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── UI CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Time‑Series Validator", layout="wide")

# ── CONSTANTS & SYNONYMS ────────────────────────────────────────────────────
MANDATORY: Dict[str, str] = {"date": "datetime64[ns]"}
INDICATORS: Dict[str, str] = {
    "gdp": "float64",
    "cpi": "float64",
    "unemployment_rate": "float64",
    "population": "int64",
}
SYN: Dict[str, List[str]] = {
    # date aliases
    "date": [
        "date", "Date", "year", "Year", "time", "Time", "period",
        "Period Year", "month", "Month", "quarter", "Quarter", "YearMonth",
    ],
    # indicator aliases
    "gdp": ["gdp_usd", "gdpcurrent"],
    "cpi": ["consumer price index"],
    "unemployment_rate": ["unemployment"],
    "population": ["pop"],
    # typo
    "Balence of Payment": ["BOP"],
}
SYN_LC: Dict[str, List[str]] = {k: [a.lower() for a in v] for k, v in SYN.items()}

FREQ_MAP: Dict[str, Tuple[str, str, pd.Period]] = {
    "Yearly":    ("AS", "Y", pd.Period("2025",   freq="Y")),
    "Quarterly": ("QS", "Q", pd.Period("2025Q2", freq="Q")),
    "Monthly":   ("MS", "M", pd.Period("2025-06", freq="M")),  # <- ASCII hyphen
}

ID_COLS_DEFAULT = {"date", "country_code", "country_name", "item", "unit"}
ALL_CANONICAL = set(MANDATORY) | set(INDICATORS)

# ── HELPER DATACLASS ────────────────────────────────────────────────────────
@dataclass
class Plan:
    rename: Dict[str, str] = field(default_factory=dict)
    header_row: int = 0   # index (0‑based) of header row

# ── LOW‑LEVEL UTILS ─────────────────────────────────────────────────────────
def slug(s: str) -> str:
    """simple slugify for column suffixes"""
    return (
        s.lower().strip()
         .replace(" ", "_").replace("(", "").replace(")", "")
    )

@st.cache_data(show_spinner=False)
def sniff(data: bytes, filename: str) -> pd.DataFrame:
    """Load CSV/Excel with no header so we can detect it."""
    ext = Path(filename).suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(io.BytesIO(data), header=None)
    return pd.read_excel(io.BytesIO(data), sheet_name=0, header=None, engine="openpyxl")

def detect_header(df: pd.DataFrame, look_ahead: int = 3) -> int:
    """First row where >50 % cells contain letters."""
    for i in range(min(look_ahead, len(df))):
        if df.iloc[i].astype(str).str.contains(r"[A-Za-z]").mean() > 0.5:
            return i
    return 0

def suggest_map(cols: Sequence[str]) -> Dict[str, str]:
    """Guess canonical names from synonyms / fuzzy matching."""
    import difflib
    mapping: Dict[str, str] = {}
    for c in cols:
        key = c.lower().strip()
        if key in ALL_CANONICAL:
            mapping[c] = key
            continue
        for tgt, aliases in SYN_LC.items():
            if key in aliases:
                mapping[c] = tgt
                break
        else:
            guess = difflib.get_close_matches(key, list(ALL_CANONICAL), n=1, cutoff=0.85)
            if guess:
                mapping[c] = guess[0]
    return mapping

def build_plan(raw: pd.DataFrame) -> Plan:
    hdr = detect_header(raw)
    rename = suggest_map(raw.iloc[hdr].astype(str))
    return Plan(rename, hdr)

def apply_plan(raw: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    """Apply header row & renames; drop empty columns."""
    df = raw.copy()
    df.columns = df.iloc[plan.header_row]
    df = df.iloc[plan.header_row + 1:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.isna()]
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    if plan.rename:
        df.rename(columns=plan.rename, inplace=True)
    return df

# ── WIDE‑TO‑LONG ────────────────────────────────────────────────────────────
def _is_year(val: Union[str, int, float]) -> bool:
    try:
        y = int(float(val))
        return 1900 <= y <= 2100
    except Exception:
        return False

def wide_to_long_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Convert classic year‑wide tables to tidy long form."""
    year_cols = [c for c in df.columns if _is_year(c)]
    if len(year_cols) < 5 or "date" in df.columns:
        return df
    id_cols = [c for c in df.columns if c not in year_cols]
    df_long = (
        df.melt(id_vars=id_cols, value_vars=year_cols,
                var_name="date", value_name="value")
          .dropna(subset=["value"])
    )
    df_long["date"] = pd.to_datetime(df_long["date"].astype(int), format="%Y")
    for alt in ("Country Code", "Areacode", "Area Code", "AreaCode"):
        if alt in df_long.columns:
            df_long.rename(columns={alt: "country_code"}, inplace=True)
            break
    return df_long.reset_index(drop=True)

# ── DOMAIN‑SPECIFIC RENAME MAP ──────────────────────────────────────────────
def build_rename_map(df: pd.DataFrame) -> Dict[str, str]:
    rename = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in {"period year", "periodyear", "period_year"}:
            rename[col] = "date"; continue
        if low in {"period value", "periodvalue", "period_value"}:
            rename[col] = "value"; continue
        if low in {"area", "area (english)", "country (english)", "country english"}:
            rename[col] = "country_name"; continue
        if "item code" in low or "indicatorcode" in low or low == "item":
            rename[col] = "item"; continue
        if low in {"unit", "element code", "element"}:
            rename[col] = "unit"; continue
        if low in {"balance of payment", "balance of payments",
                   "balence of payment", "bop"}:
            rename[col] = "value"; continue
        for tgt in INDICATORS:
            if low == tgt or low in SYN_LC.get(tgt, []):
                rename[col] = tgt; break
    deduped = {}
    for src, canon in rename.items():
        if canon == "value":
            deduped[src] = "value"
        else:
            new = f"{canon}_{slug(src)}"
            base, i = new, 1
            while new in deduped.values():
                i += 1; new = f"{base}_{i}"
            deduped[src] = new
    return deduped

# ── AUTO‑COERCE NUMERIC OBJECT COLS ─────────────────────────────────────────
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col == "date" or df[col].dtype != "object":
            continue
        sample = df[col].dropna().astype(str).str.replace(",", "").str.strip()
        if sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.5:
            df[col] = pd.to_numeric(sample, errors="coerce")
    return df

# ── NUMERIC SERIES DETECTION ────────────────────────────────────────────────
def numeric_series(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in ID_COLS_DEFAULT and pd.api.types.is_numeric_dtype(df[c])
    ]

# ── ORDER‑PRESERVING DE‑DUPLICATION ─────────────────────────────────────────
def dedupe(seq: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📊 Time‑Series Data Validator")

    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
    if file is None:
        st.info("👆 Upload a file to begin.")
        return

    # ── INGEST & CLEAN ────────────────────────────────────────────────────
    raw   = sniff(file.getvalue(), file.name)
    plan  = build_plan(raw)
    clean = apply_plan(raw, plan)
    clean = wide_to_long_if_needed(clean)
    clean = coerce_numeric(clean)

    extra = build_rename_map(clean)
    if extra:
        clean = (clean
                 .rename(columns=extra)
                 .loc[:, lambda d: ~d.columns.duplicated()])

    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    else:
        st.error("❌ No 'date' column detected. Adjust synonyms or file.")
        return

    # debug helper
    with st.expander("Column dtypes", False):
        st.write(clean.dtypes)

    series_cols = numeric_series(clean)
    if not series_cols:
        st.error("No numeric series found. "
                 "Check the dtypes above – object columns "
                 "should have been converted automatically.")
        return

    # ── GROUPING UI ───────────────────────────────────────────────────────
    all_cols      = list(clean.columns)
    default_group = [c for c in ("country_code", "country_name", "item", "unit")
                     if c in all_cols]
    st.markdown("### Series grouping")
    picked      = st.multiselect("Group by", all_cols, default=default_group)
    group_cols  = dedupe(picked) or [None]

    # iterator over groups
    def iter_groups(df: pd.DataFrame):
        if group_cols and group_cols[0] is not None:
            yield from df.groupby(group_cols)
        else:
            yield ((), df)

    # ── VIEW SELECTOR ─────────────────────────────────────────────────────
    view = st.selectbox(
        "Choose view",
        ["Preview", "Gap analysis", "Outlier scan", "Structural break", "Custom rules"],
        index=0
    )

    # ────────────────────────────────────────────────────────────────────
    if view == "Preview":
        st.dataframe(clean.head())

    # ── GAP ANALYSIS ────────────────────────────────────────────────────
    elif view == "Gap analysis":
        st.subheader("Gap Analysis")
        freq_label = st.selectbox("Frequency", list(FREQ_MAP))
        drange, pcode, fixed_end = FREQ_MAP[freq_label]

        chosen = st.multiselect("Series to scan", series_cols, default=series_cols)
        if not chosen:
            st.info("Select at least one series."); return

        def fmt(p: pd.Period) -> str:
            if p.freqstr == "Y":
                return str(p.year)
            if p.freqstr == "Q":
                return f"{p.year}Q{p.quarter}"
            if p.freqstr == "M":
                return p.strftime("%b %Y")
            return str(p)

        rows = []
        for name, grp in iter_groups(clean):
            grp_id = ", ".join(f"{c}={v}" for c, v in zip(group_cols, name)) \
                     if name else "All data"
            per = grp["date"].dt.to_period(pcode)
            start_p, end_p = per.min(), fixed_end
            full_idx = pd.period_range(start_p, end_p, freq=pcode)
            missing  = full_idx.difference(per)
            for col in chosen:
                rows.append({
                    "Series":    grp_id,
                    "Indicator": col,
                    "Range":     f"{fmt(start_p)}–{fmt(end_p)}",
                    "Missing":   [fmt(p) for p in missing],
                })
        st.dataframe(pd.DataFrame(rows))

    # ── OUTLIER SCAN ────────────────────────────────────────────────────
    elif view == "Outlier scan":
        st.subheader("Outlier Scan (z‑score)")
        threshold = st.slider("Z‑score threshold", 1.0, 5.0, 3.0)
        var       = st.selectbox("Series", series_cols)

        outliers = []
        for name, grp in iter_groups(clean):
            label = ", ".join(f"{c}={v}" for c, v in zip(group_cols, name)) \
                    if name else "All data"
            vals = grp[var].dropna()
            if vals.empty:
                continue
            mean, std = vals.mean(), vals.std(ddof=0) or 1
            z = (grp[var] - mean) / std
            for idx in grp.index[(z.abs() > threshold)]:
                outliers.append({
                    "group":   name,
                    "Series":  label,
                    "Variable": var,
                    "Date":    grp.at[idx, "date"],
                    "Value":   grp.at[idx, var],
                    "Z":       float(z.at[idx]),
                })

        if not outliers:
            st.info(f"No outliers beyond ±{threshold}."); return

        df_o = pd.DataFrame(outliers)
        st.dataframe(df_o)
        lbl = {i: f"{i}: {r['Series']} | {var} @ {r['Date'].date()}"
               for i, r in df_o.iterrows()}
        sel_idx = st.selectbox("Plot which one?", list(lbl), format_func=lbl.get)
        sel     = df_o.loc[sel_idx]

        # plot
        if group_cols[0] is None:
            df_plot = clean.copy()
        else:
            mask = (clean[group_cols] == pd.Series(sel["group"]).values).all(axis=1)
            df_plot = clean[mask]
        df_plot = df_plot.sort_values("date")
        smooth  = df_plot[var].rolling(5, min_periods=1, center=True).mean()

        fig, ax = plt.subplots()
        ax.scatter(df_plot["date"], df_plot[var], s=20, alpha=0.6)
        ax.plot(df_plot["date"], smooth, linewidth=2)
        ax.scatter(sel["Date"], sel["Value"], s=120, c="red",
                   edgecolors="black", zorder=3)
        ax.set_title(lbl[sel_idx]); ax.set_xlabel("Date"); ax.set_ylabel(var)
        st.pyplot(fig)

    # ── STRUCTURAL BREAK (2nd diff) ─────────────────────────────────────
    elif view == "Structural break":
        st.subheader("Structural break – 2nd difference")
        var   = st.selectbox("Series", series_cols)
        std   = float(clean[var].std(ddof=0) or 0)
        thresh = st.slider("Δ² threshold", 0.0, max(std*2, 1.0), std if std>0 else 0.1)

        kinks = []
        for name, grp in iter_groups(clean):
            label = ", ".join(f"{c}={v}" for c, v in zip(group_cols, name)) \
                    if name else "All data"
            s     = grp.sort_values("date")[var].astype(float)
            d2    = s.diff().diff().abs()
            for idx in d2.index[d2 > thresh]:
                kinks.append({
                    "group":  name,
                    "Series": label,
                    "Variable": var,
                    "Date":   grp.loc[idx, "date"],
                    "Δ²":     float(d2.loc[idx]),
                })

        if not kinks:
            st.info("No kinks above threshold."); return

        df_k = pd.DataFrame(kinks)
        st.dataframe(df_k)
        lbl = {i: f"{i}: {r['Series']} | {var} @ {r['Date'].date()} (Δ²={r['Δ²']:.2f})"
               for i, r in df_k.iterrows()}
        sel_idx = st.selectbox("Plot which one?", list(lbl), format_func=lbl.get)
        row     = df_k.loc[sel_idx]

        if group_cols[0] is None:
            df_p = clean.copy()
        else:
            mask = (clean[group_cols] == pd.Series(row["group"]).values).all(axis=1)
            df_p = clean[mask]

        df_p   = df_p.sort_values("date")
        smooth = df_p[var].rolling(5, min_periods=1, center=True).mean()

        fig, ax = plt.subplots()
        ax.scatter(df_p["date"], df_p[var], s=20)
        ax.plot(df_p["date"], smooth, linewidth=2)
        ax.axvline(row["Date"], linestyle="--", c="red")
        ax.set_title(lbl[sel_idx]); ax.set_xlabel("Date"); ax.set_ylabel(var)
        st.pyplot(fig)

    # ── CUSTOM RULES ────────────────────────────────────────────────────
    else:
        st.subheader("Custom rules")
        num_cols = numeric_series(clean)

        if "rules" not in st.session_state:
            st.session_state.rules = []

        if st.session_state.rules:
            st.markdown("#### Existing rules")
            for i, r in enumerate(st.session_state.rules):
                scope = f" ({r['start']}→{r['end']})" if r.get("start") else ""
                st.write(f"**{i}**: {r['col']} {r['op']} {r['val']}{scope}")
            remove = st.multiselect("Remove rules", list(range(len(st.session_state.rules))))
            if st.button("Delete selected") and remove:
                for i in sorted(remove, reverse=True):
                    st.session_state.rules.pop(i)
                st.success("Rules removed.")
        else:
            st.info("No rules yet.")

        with st.form("add_rule"):
            new_col = st.selectbox("Column", num_cols)
            new_op  = st.selectbox("Operator", ["<=", ">=", "<", ">", "==", "!="])
            new_val = st.number_input("Value", value=0.0)
            scoped  = st.checkbox("Restrict to date range?")
            start = end = None
            if scoped:
                start = st.date_input("Start", value=clean["date"].min().date())
                end   = st.date_input("End",   value=clean["date"].max().date(),
                                      min_value=start)
            if st.form_submit_button("Add"):
                st.session_state.rules.append({
                    "col": new_col, "op": new_op, "val": new_val,
                    "start": start, "end": end,
                })
                st.success("Rule added.")

        # evaluate
        if st.session_state.rules:
            viol_list = []
            for idx, r in enumerate(st.session_state.rules):
                df_sub = clean.copy()
                if r["start"]:
                    mask = (df_sub["date"] >= pd.to_datetime(r["start"])) & \
                           (df_sub["date"] <= pd.to_datetime(r["end"]))
                    df_sub = df_sub[mask]
                expr = {
                    "<=": df_sub[r["col"]] <= r["val"],
                    ">=": df_sub[r["col"]] >= r["val"],
                    "<":  df_sub[r["col"]] <  r["val"],
                    ">":  df_sub[r["col"]] >  r["val"],
                    "==": df_sub[r["col"]] == r["val"],
                    "!=": df_sub[r["col"]] != r["val"],
                }[r["op"]]
                bad = df_sub[~expr]
                if not bad.empty:
                    bad = bad.assign(rule_id=idx,
                                     rule=f"{r['col']} {r['op']} {r['val']}")
                    viol_list.append(bad)
            if viol_list:
                st.markdown("#### Violations")
                st.dataframe(pd.concat(viol_list, ignore_index=True))
            else:
                st.success("✅ All data satisfy the rules.")

    # ── DOWNLOAD ────────────────────────────────────────────────────────
    csv = clean.to_csv(index=False).encode()
    st.sidebar.download_button("Download cleaned CSV", csv,
                               "cleaned.csv", "text/csv")

if __name__ == "__main__":
    main()
