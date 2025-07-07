# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:24:03 2025

@author: LWu

Time‑Series Validator

• Converts wide year‑columns → long format automatically
• Works on Streamlit ≥ 1.2

To run this file:
pip install all packages below
streamlit run data_validation_prototype.py
"""

import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import ruptures as rpt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time‑Series Validator", layout="wide")

# ──────────────────────────────────────────────────────────────────────
# Expected columns & synonyms
# ──────────────────────────────────────────────────────────────────────
MANDATORY = {"date": "datetime64[ns]"}
INDICATORS = {
    "gdp": "float64",
    "cpi": "float64",
    "unemployment_rate": "float64",
    "population": "int64",
}
SYN = {
    "date": ["year", "time", "period", "Period Year"],
    "gdp": ["gdp_usd", "gdpcurrent"],
    "cpi": ["consumer price index"],
    "unemployment_rate": ["unemployment"],
    "population": ["pop"],
    "Balence of Payment": ["BOP"],
}

# ─── Frequency mapping for Gap Analysis ─────────────────────────────────
# Keys shown in the UI; first element is the pandas DateOffset string
# (used by pd.date_range), second is a dummy/period code (not used here).
freq_map = {
    "Daily":     ("D",   "D"),
    "Monthly":   ("MS",  "M"),
    "Quarterly": ("QS",  "Q"),
    "Yearly":    ("AS",  "A"),
}
# ─────────────────────────────────────────────────────────────────────────


# lower-case aliases for synonym matching
SYN_LC = {tgt: [alias.lower() for alias in aliases]
          for tgt, aliases in SYN.items()}


ALL = set(MANDATORY) | set(INDICATORS)

# ──────────────────────────────────────────────────────────────────────
# Helper dataclass
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Plan:
    rename: Dict[str, str] = field(default_factory=dict)
    header_row: int = 0  # always an int so plan is never "empty"

# ──────────────────────────────────────────────────────────────────────
# Ingest helpers
# ──────────────────────────────────────────────────────────────────────

def sniff(bytes_, name):
    ext = Path(name).suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(io.BytesIO(bytes_), header=None)
    return pd.read_excel(io.BytesIO(bytes_), sheet_name=0, header=None, engine="openpyxl")


def detect_header(df: pd.DataFrame) -> int:
    for i in range(3):
        text_ratio = df.iloc[i].astype(str).str.contains("[A-Za-z]").mean()
        if text_ratio > 0.5:
            return i
    return 0


def suggest_map(cols: List[str]) -> Dict[str, str]:
    import difflib
    mapping: Dict[str, str] = {}
    for c in cols:
        key = c.lower().strip()
        # direct match to known column names
        if key in ALL:
            mapping[c] = key
            continue
        # match via lower-cased synonyms
        for tgt, aliases in SYN_LC.items():
            if key in aliases:
                mapping[c] = tgt
                break
        else:
            # fallback: fuzzy match against the canonical list
            maybe = difflib.get_close_matches(key, list(ALL), n=1, cutoff=0.85)
            if maybe:
                mapping[c] = maybe[0]
    return mapping


def build_plan(df: pd.DataFrame) -> Plan:
    hdr = detect_header(df)
    rename = suggest_map(df.iloc[hdr].astype(str))
    return Plan(rename, hdr)


def apply_plan(df: pd.DataFrame, p: Plan) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.iloc[p.header_row]
    df = df.iloc[p.header_row + 1:]
    # Drop completely blank / unnamed columns which confuse Arrow
    df = df.loc[:, ~df.columns.isna()]
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    if p.rename:
        df.rename(columns=p.rename, inplace=True)
    return df.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────
# Wide‑to‑long detector
# ──────────────────────────────────────────────────────────────────────

def _is_year(val):
    try:
        yr = int(float(val))
        return 1900 <= yr <= 2100
    except Exception:
        return False


def wide_to_long_if_needed(df: pd.DataFrame) -> pd.DataFrame:
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
    # Drop rows where country_code (if present) is blank
    if "country_code" in df_long.columns:
        df_long = df_long[df_long["country_code"].astype(str).str.strip() != ""]
    return df_long.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────

st.title("📊 Time‑Series Data Validator")
file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls", "txt"])
if file is None:
    st.stop()

raw_df = sniff(file.getvalue(), file.name)
plan = build_plan(raw_df)
st.markdown("### Auto‑detected fixes")
st.json({"header_row": plan.header_row, "rename": plan.rename})

clean = apply_plan(raw_df, plan)
clean = wide_to_long_if_needed(clean)

# ── Handle missing country_code by falling back to a name column ──
if "country_code" not in clean.columns:
    # look for any reasonable country-name field
    for alt in ["Country English", "Country Eng", "Country", "country", "country_name", "Country (English)"]:
        if alt in clean.columns:
            clean = clean.rename(columns={alt: "country_name"})
            break



# Ensure 'date' is proper datetime
if "date" in clean.columns and not np.issubdtype(clean["date"].dtype, np.datetime64):
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")

# ── Numeric coercion ───────────────────────────────────────────
# Only coerce the canonical "value" column or obviously numeric dtypes;
# leave ID/text columns like country_code untouched.
if "value" in clean.columns:
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
else:
    # ── Coerce numeric series ──────────────────────────────────────────────
# Identify ID columns we want to keep as strings
    id_cols = {"date", "country_code", "country_name"}
    for c in clean.columns:
        if c not in id_cols:
            clean[c] = pd.to_numeric(clean[c], errors="coerce")

# ───────────────────────────────────────────────────────────────────
#  View selector & group-aware analytics
# ───────────────────────────────────────────────────────────────────
view = st.selectbox(
    "Choose view",
    ["Preview", "Gap analysis", "Outlier scan", "Structural break", "Custom rules"],
    index=0
)

# Determine grouping
# Determine grouping (fall back to country_name if needed)
# ── Build grouping key ────────────────────────────────────────────────
group_cols = []
# pick either code or name
if "country_code" in clean.columns:
    group_cols.append("country_code")
elif "country_name" in clean.columns:
    group_cols.append("country_name")
# then any sub-series dimensions
# ── Flexible detection of “item” and “unit” columns ────────────────────────
import difflib

# Canonical keys → lists of aliases (all matched case-insensitive)
GROUP_SYNS: dict[str, list[str]] = {
    "item": [
        "item", "indicator", "series", "measure", "product", "metric", "crop",
        "category", "type", "subgroup"
    ],
    "unit": [
        "unit", "units", "measure unit", "uom", "uoms", "measurement",
        "denominator"
    ],
}

# For each canonical role, scan your actual columns
for role, aliases in GROUP_SYNS.items():
    # first, exact (lower-case) match
    matched = False
    for col in clean.columns:
        if col.lower() in {a.lower() for a in aliases}:
            group_cols.append(col)
            matched = True
            break
    if matched:
        continue

    # fuzzy fallback: pick the best alias match above a threshold
    for col in clean.columns:
        choice = difflib.get_close_matches(col.lower(), [a.lower() for a in aliases],
                                          n=1, cutoff=0.75)
        if choice:
            group_cols.append(col)
            break
# ───────────────────────────────────────────────────────────────────────────

# fallback to “all data” if nothing to group by
if not group_cols:
    group_cols = [None]
# ──────────────────────────────────────────────────────────────────────

if view == "Preview":
    st.subheader("Cleaned Data Preview")
    st.dataframe(clean.head())

elif view == "Gap analysis":
    st.markdown("## Gap Analysis")

    # 1. Define allowed frequencies and defaults
    freq_map = {
        "Yearly": ("AS", "A", pd.Period("2025", freq="A")),
        "Quarterly":("QS", "Q", pd.Period("2025Q2", freq="Q")),
        "Monthly":  ("MS", "M", pd.Period("2025-06", freq="M")),
    }
    opts = list(freq_map.keys())
    freq_label = st.selectbox("Frequency", opts, index=0)  # Yearly default

    fetch, pcode, fixed_end = freq_map[freq_label]

    def fmt_period(p: pd.Period) -> str:
        if p.freq == "A":
            return str(p.year)
        if p.freq == "Q":
            return f"{p.year}Q{p.quarter}"
        if p.freq == "M":
            return p.strftime("%b %Y")
        return str(p)

    results = []
    groups = (clean.groupby(group_cols) 
              if group_cols[0] is not None 
              else [((), clean)])

    for name, grp in groups:
        # identifier
        if isinstance(name, tuple):
            identifier = ", ".join(f"{col}={val}"
                                   for col, val in zip(group_cols, name))
        elif name is None:
            identifier = "All data"
        else:
            identifier = f"{group_cols[0]}={name}"

        # build period series
        dates = pd.to_datetime(grp["date"], errors="coerce")
        start_p = dates.min().to_period(pcode)
        end_p   = fixed_end

        # full index and missing detection
        all_periods = pd.period_range(start_p, end_p, freq=pcode)
        existing   = dates.dt.to_period(pcode).unique()
        missing    = all_periods.difference(existing)

        results.append({
            "Series":         identifier,
            "Range":          f"{fmt_period(start_p)}–{fmt_period(end_p)}",
            "Missing":        [fmt_period(p) for p in missing],
        })

    st.dataframe(pd.DataFrame(results))




elif view == "Outlier scan":
    st.markdown("## Outlier Scan")
    st.caption("Method: Z-score = (value – mean) / std; flags |z| > threshold")
    threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)

    # detect numeric series columns
    id_cols = {"date", "country_code", "country_name", "item", "unit"}
    num_cols = [c for c in clean.columns
                if c not in id_cols and pd.api.types.is_numeric_dtype(clean[c])]

    outliers = []
    if group_cols[0] is not None:
        grouped = clean.groupby(group_cols)
    else:
        grouped = [((), clean)]

    for name, grp in grouped:
        # same identifier logic as above
        if isinstance(name, tuple):
            identifier = ", ".join(f"{col}={val}"
                                   for col, val in zip(group_cols, name))
        elif name is None:
            identifier = "All data"
        else:
            identifier = f"{group_cols[0]}={name}"

        for col in num_cols:
            vals = grp[col].dropna()
            if vals.empty:
                continue
            mean, std = vals.mean(), vals.std(ddof=0)
            z = (grp[col] - mean) / std
            mask = z.abs() > threshold

            for i in grp.index[mask]:
                outliers.append({
                    "group":    name,           # ← carry the grouping key
                    "Series":   identifier,
                    "Variable": col,
                    "Date":     grp.at[i, "date"],
                    "Value":    grp.at[i, col],
                    "Z-score":  float(z.at[i]),
        })

    if outliers:
        df_o = pd.DataFrame(outliers)
        st.dataframe(df_o)

        # Build user-friendly labels
        labels = [
            f"{idx}: {row['Series']} | {row['Variable']} @ {row['Date'].date()}"
            for idx, row in df_o.iterrows()
            ]
        choice = st.selectbox("Select outlier to plot", labels)

        # Extract the chosen row
        sel_idx = int(choice.split(":", 1)[0])
        sel = df_o.loc[sel_idx]

        # Filter original data to that group & variable
        if group_cols[0] is not None:
            grp_name = sel["group"]
            if isinstance(grp_name, tuple):
                mask = np.ones(len(clean), dtype=bool)
                for col, val in zip(group_cols, grp_name):
                    mask &= (clean[col] == val)
            else:
                mask = (clean[group_cols[0]] == grp_name)
            df_plot = clean[mask]
        else:
            df_plot = clean

        # Ensure date is datetime
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
        var = sel["Variable"]

        # Draw line and highlight the outlier
        fig, ax = plt.subplots()
        ax.plot(df_plot["date"], df_plot[var], label=var)
        ax.scatter(
            sel["Date"], sel["Value"],
            color="red", zorder=5, label="Outlier"
            )
        ax.set_title(f"{sel['Series']} – {var}")
        ax.set_xlabel("Date")
        ax.set_ylabel(var)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info(f"No outliers found at |z| > {threshold}.")

elif view == "Structural break":
    st.markdown("## Structural Break Detection")
    st.caption(
        "Method: PELT (L2). Statistic = SSE reduction " +
        "(sum of squared residuals drop at the breakpoint)."
    )
    pen = st.slider("Penalty for breakpoint detection", 1.0, 10.0, 3.0)

    # Identify numeric series columns (as before)
    id_cols = {"date", "country_code", "country_name", "item", "unit"}
    num_cols = [
        c for c in clean.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(clean[c])
    ]

    breaks = []
    groups = clean.groupby(group_cols) if group_cols[0] is not None else [((), clean)]

    for name, grp in groups:
        # build a human-readable identifier
        if isinstance(name, tuple):
            series_id = ", ".join(f"{col}={val}"
                                  for col, val in zip(group_cols, name))
        elif name is None:
            series_id = "All data"
        else:
            series_id = f"{group_cols[0]}={name}"

        # convert dates & values
        dates = pd.to_datetime(grp["date"], errors="coerce")
        for col in num_cols:
            vals = grp[col].dropna().values
            if len(vals) < 10:
                continue  # too short to detect
            # run PELT to find one breakpoint
            algo = rpt.Pelt(model="l2").fit(vals)
            # the last index returned is len(vals); drop it
            bkps = algo.predict(pen=pen)[:-1]
            for b in bkps:
                # locate the original row and values
                orig_idx = grp.dropna(subset=[col]).index[b]
                b_date   = dates.iloc[b]
                b_val    = grp.at[orig_idx, col]
                prev_val = grp[col].iloc[b-1] if b > 0 else np.nan
                change   = (b_val - prev_val) if not np.isnan(prev_val) else np.nan

    # compute sum-of-squared-errors reduction
                vals = grp[col].dropna().values
                mean_all = vals.mean()
                sse_all  = ((vals - mean_all) ** 2).sum()
                sse1     = ((vals[:b]   - vals[:b].mean()) ** 2).sum()
                sse2     = ((vals[b:]   - vals[b:].mean()) ** 2).sum()
                stat     = float(sse_all - (sse1 + sse2))

            breaks.append({
                "group":      name,
                "Series":     series_id,
                "Variable":   col,
                "Breakpoint": b_date,
                "Statistic":  stat,
                "Change":     float(change),
    })


    # ── Breakpoint plotting ───────────────────────────────────────────────
    if breaks:
        df_b = pd.DataFrame(breaks)
        st.dataframe(df_b)                # ← Insert this line here
    else:
        st.info("No structural breaks detected with the given penalty.")
    
    if breaks:
        df_b = pd.DataFrame(breaks)

    # build labels for user
        labels = [
            f"{idx}: {row['Series']} | {row['Variable']} @ {row['Breakpoint'].date()}"
            for idx, row in df_b.iterrows()]
        sel = st.selectbox("Select breakpoint to plot", labels)

    # extract the chosen row
        chosen_idx = int(sel.split(":", 1)[0])
        chosen = df_b.loc[chosen_idx]

    # filter the original clean DataFrame to that group & variable
        if group_cols[0] is not None:
        # unpack the tuple or single key
            name = chosen["group"]
            if isinstance(name, tuple):
                mask = np.ones(len(clean), dtype=bool)
                for col, val in zip(group_cols, name):
                    mask &= (clean[col] == val)
            else:
                mask = (clean[group_cols[0]] == name)
            df_plot = clean[mask]
        else:
            df_plot = clean

        var = chosen["Variable"]
        # ensure date is datetime
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")

        # plot
        fig, ax = plt.subplots()
        ax.plot(df_plot["date"], df_plot[var])
        ax.axvline(chosen["Breakpoint"], linestyle="--")
        ax.set_title(f"{chosen['Series']} – {var}")
        ax.set_xlabel("Date")
        ax.set_ylabel(var)
        st.pyplot(fig)

    else:
        st.info("No structural breaks detected with the given penalty.")
        
elif view == "Custom rules":
    st.markdown("## Custom Rules")
    st.caption(
        "Define one or more rules of the form “column OP value” "
        "(e.g. value <= 10000), optionally scoped to a date range."
    )

    # 1. Prepare list of numeric columns
    id_cols = {"date", "country_code", "country_name", "item", "unit"}
    num_cols = [
        c for c in clean.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(clean[c])
    ]

    # 2. Initialize session-state
    if "rules" not in st.session_state:
        st.session_state.rules = []

    # 3. Show existing rules
    if st.session_state.rules:
        st.markdown("### Current rules")
        for i, r in enumerate(st.session_state.rules):
            scope = (
                f" between {r['start']} and {r['end']}"
                if r.get("start") else ""
            )
            st.write(f"**{i}**: `{r['col']} {r['op']} {r['val']}`{scope}")

        # ── Removal UI ──────────────────────────────────────────
        to_remove = st.multiselect(
            "Select rule IDs to remove",
            options=list(range(len(st.session_state.rules))),
            format_func=lambda x: f"{x}: {st.session_state.rules[x]['col']} "
                                  f"{st.session_state.rules[x]['op']} "
                                  f"{st.session_state.rules[x]['val']}"
        )
        if st.button("Remove selected rules") and to_remove:
            # Remove highest indices first to avoid reindexing issues
            for idx in sorted(to_remove, reverse=True):
                st.session_state.rules.pop(idx)
            st.success(f"Removed rules: {to_remove}")
    else:
        st.write("No rules defined yet.")

    # 4. Form to add a new rule
    with st.form("add_rule"):
        st.selectbox("Column", num_cols, key="new_col")
        st.selectbox("Operator",
                     ["<=", ">=", "<", ">", "==", "!="],
                     key="new_op")
        st.number_input("Value", value=0.0, key="new_val")
        scope = st.checkbox("Restrict to a date range?", key="new_scope")
        if scope:
            new_start = st.date_input(
                "Start date", key="new_start",
                min_value=clean["date"].min().date(),
                max_value=clean["date"].max().date()
            )
            new_end = st.date_input(
                "End date", key="new_end",
                min_value=new_start,
                max_value=clean["date"].max().date()
            )
        else:
            new_start = new_end = None

        if st.form_submit_button("Add rule"):
            st.session_state.rules.append({
                "col":   st.session_state.new_col,
                "op":    st.session_state.new_op,
                "val":   st.session_state.new_val,
                "start": new_start,
                "end":   new_end,
            })

    # 5. Evaluate rules and collect violations
    if st.session_state.rules:
        violations = []
        for idx, r in enumerate(st.session_state.rules):
            df_sub = clean.copy()
            if r["start"]:
                df_sub = df_sub[
                    (df_sub["date"] >= pd.to_datetime(r["start"])) &
                    (df_sub["date"] <= pd.to_datetime(r["end"]))
                ]
            expr = {
                "<=": df_sub[r["col"]] <= r["val"],
                ">=": df_sub[r["col"]] >= r["val"],
                "<":  df_sub[r["col"]] <  r["val"],
                ">":  df_sub[r["col"]] >  r["val"],
                "==": df_sub[r["col"]] == r["val"],
                "!=": df_sub[r["col"]] != r["val"],
            }[r["op"]]
            viol = df_sub[~expr]
            if not viol.empty:
                viol = viol.copy()
                viol["rule_id"] = idx
                viol["rule"]    = f"{r['col']} {r['op']} {r['val']}"
                if r["start"]:
                    viol["start_date"] = r["start"]
                    viol["end_date"]   = r["end"]
                violations.append(viol)

        if violations:
            df_v = pd.concat(violations, ignore_index=True)
            st.markdown("### Violations")
            st.dataframe(df_v)
        else:
            st.success("👍 All data points satisfy your rules.")




# ───────────────────────────────────────────────────────────────────
# Download cleaned dataset
# ───────────────────────────────────────────────────────────────────
_ = st.sidebar.download_button(
    label="Download cleaned CSV",
    data=clean.to_csv(index=False),
    file_name="cleaned.csv",
    mime="text/csv"
)


