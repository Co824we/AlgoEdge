
import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# ALGO Edge Performance History
# Portfolio + Strategy Monte Carlo
# ============================================================

st.set_page_config(
    page_title="ALGO Edge Performance History",
    page_icon="📈",
    layout="wide",
)

st.title("ALGO Edge Performance History")
st.caption("Portfolio balance-history analysis, individual strategy analysis, sit-out overlays, and Monte Carlo projections.")

st.markdown(
    """
This build supports the broker-style portfolio balance-history export with columns like
`Date`, `NLV`, `Day_PL`, `Day_PL_Percent`, `Deposits/Withdrawals`, and `AllTime_PL_Percent`.

It also preserves individual strategy analysis for CSVs with a date column and either a
strategy column, balance column, return column, or P/L column.
"""
)


# ------------------------------------------------------------
# Column detection
# ------------------------------------------------------------

DATE_CANDIDATES = [
    "date", "Date", "DATE", "datetime", "Datetime", "timestamp", "Timestamp",
    "time", "Time", "close_time", "Close Time", "Trade Date", "trade_date",
    "entry_date", "Entry Date", "exit_date", "Exit Date",
]

BALANCE_CANDIDATES = [
    "NLV", "nlv", "Net Liquidation Value", "net liquidation value",
    "NetLiq", "netliq", "net_liq", "Net Liq", "NetLiquidation", "Net Liquidation",
    "balance", "Balance", "BALANCE", "equity", "Equity", "account_value", "Account Value",
    "value", "Value", "cumulative_balance", "Cumulative Balance", "ending_balance", "Ending Balance",
]

DAY_PNL_CANDIDATES = [
    "Day_PL", "day_pl", "Day P/L", "Daily P/L", "Daily PL", "daily_pl",
    "Day PnL", "Daily PnL", "day_pnl", "daily_pnl",
]

CASHFLOW_CANDIDATES = [
    "Deposits/Withdrawals", "deposits/withdrawals", "Deposits Withdrawals",
    "Deposit/Withdrawal", "deposit/withdrawal", "Cash Flow", "cash_flow",
    "CashFlow", "Net Deposits", "net_deposits", "Deposits", "deposits",
    "Withdrawals", "withdrawals",
]

RETURN_CANDIDATES = [
    "return", "Return", "returns", "Returns", "daily_return", "Daily Return",
    "pct_return", "Pct Return", "% Return", "percent_return", "Percent Return",
    "Day_PL_Percent", "day_pl_percent", "Day P/L Percent", "Daily P/L Percent",
    "Daily PL Percent",
]

ALLTIME_PCT_CANDIDATES = [
    "AllTime_PL_Percent", "AllTime P/L Percent", "AllTime PL Percent",
    "All Time P/L Percent", "All Time PL Percent", "AllTime Percent",
]

ALLTIME_PL_CANDIDATES = [
    "AllTime_PL", "AllTime P/L", "AllTime PL", "All Time P/L", "All Time PL",
]

PNL_CANDIDATES = [
    "pnl", "PnL", "P/L", "p/l", "profit", "Profit", "daily_pnl", "Daily PnL",
    "net_profit", "Net Profit", "Net P/L", "Realized P/L", "realized_pnl",
    "Realized PnL", "Trade P/L", "Trade PL", "trade_pnl",
]

STRATEGY_CANDIDATES = [
    "strategy", "Strategy", "STRATEGY", "system", "System", "setup", "Setup",
    "algo", "Algo", "name", "Name",
]


@dataclass
class Scope:
    label: str
    kind: str
    source_file: str
    data: pd.DataFrame
    notes: List[str]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    exact = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    normalized_candidates = [normalize_col_name(c) for c in candidates]
    for col in df.columns:
        normalized_col = normalize_col_name(col)
        for candidate in normalized_candidates:
            if candidate and normalized_col == candidate:
                return col

    for col in df.columns:
        normalized_col = normalize_col_name(col)
        for candidate in normalized_candidates:
            if candidate and candidate in normalized_col:
                return col

    return None


def to_numeric(series) -> pd.Series:
    if isinstance(series, (int, float)):
        return pd.Series(series)

    s = pd.Series(series)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    return pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def read_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.BytesIO(content), engine="python")


def safe_stem(filename: str) -> str:
    stem = str(filename).split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
    return stem.replace("_", " ").replace("-", " ").strip() or "Uploaded File"


def dedupe(label: str, existing: Dict[str, Scope]) -> str:
    if label not in existing:
        return label
    i = 2
    while f"{label} ({i})" in existing:
        i += 1
    return f"{label} ({i})"


def normalize_return_column(series: pd.Series, column_name: str = "") -> pd.Series:
    values = to_numeric(series)
    name = str(column_name).lower()
    median_abs = values.abs().median(skipna=True)

    # Columns explicitly labeled percent are percentage points:
    # 0.45 means 0.45%, not 45%.
    if "percent" in name or "%" in name:
        return values / 100.0

    # Generic return columns may be decimals or percentages.
    if pd.notna(median_abs) and median_abs > 1:
        return values / 100.0

    return values


# ------------------------------------------------------------
# Parsing
# ------------------------------------------------------------

def make_portfolio_scope(
    df: pd.DataFrame,
    filename: str,
    date_col: str,
    balance_col: str,
) -> Scope:
    notes: List[str] = []

    day_pnl_col = find_column(df, DAY_PNL_CANDIDATES)
    cashflow_col = find_column(df, CASHFLOW_CANDIDATES)
    alltime_pct_col = find_column(df, ALLTIME_PCT_CANDIDATES)
    alltime_pl_col = find_column(df, ALLTIME_PL_CANDIDATES)
    return_col = find_column(df, RETURN_CANDIDATES)

    w = df.copy()
    w["date"] = pd.to_datetime(w[date_col], errors="coerce").dt.tz_localize(None)
    w["nlv"] = to_numeric(w[balance_col])
    w["cash_flow"] = to_numeric(w[cashflow_col]) if cashflow_col else 0.0
    w["day_pl"] = to_numeric(w[day_pnl_col]) if day_pnl_col else np.nan
    w["alltime_pl"] = to_numeric(w[alltime_pl_col]) if alltime_pl_col else np.nan
    w["alltime_pl_percent"] = to_numeric(w[alltime_pct_col]) if alltime_pct_col else np.nan
    w["return_col_value"] = normalize_return_column(w[return_col], return_col) if return_col else np.nan

    w = w.dropna(subset=["date", "nlv"]).sort_values("date").copy()

    # One row per date. Cash flows and P/L are summed. NLV and cumulative metrics are last.
    d = (
        w.assign(_date=w["date"].dt.normalize())
        .groupby("_date", as_index=False)
        .agg(
            nlv=("nlv", "last"),
            cash_flow=("cash_flow", "sum"),
            day_pl=("day_pl", "sum"),
            alltime_pl=("alltime_pl", "last"),
            alltime_pl_percent=("alltime_pl_percent", "last"),
            return_col_value=("return_col_value", "last"),
        )
        .rename(columns={"_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    d["cum_cash_flow"] = d["cash_flow"].fillna(0.0).cumsum()
    d["adjusted_equity"] = d["nlv"] - d["cum_cash_flow"]

    # Return basis 1: adjusted equity. This removes deposits/withdrawals from the curve.
    d["return_adjusted_equity"] = d["adjusted_equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Return basis 2: day P/L over prior adjusted equity. Similar to adjusted equity, but explicit P/L based.
    prior_adjusted = d["adjusted_equity"].shift(1)
    d["return_daypl_prior_adjusted_equity"] = d["day_pl"] / prior_adjusted
    d.loc[prior_adjusted.isna() | (prior_adjusted == 0), "return_daypl_prior_adjusted_equity"] = 0.0
    d["return_daypl_prior_adjusted_equity"] = d["return_daypl_prior_adjusted_equity"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Return basis 3: broker cumulative percent index if available.
    if d["alltime_pl_percent"].notna().any():
        broker_index = 1 + (d["alltime_pl_percent"].fillna(method="ffill").fillna(0.0) / 100.0)
        d["return_broker_alltime_percent"] = broker_index.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        d["return_broker_alltime_percent"] = np.nan

    # Return basis 4: day P/L over prior raw NLV. Conservative for total-account return; may look flat if NLV contains idle capital.
    prior_nlv = d["nlv"].shift(1)
    d["return_daypl_prior_nlv"] = d["day_pl"] / prior_nlv
    d.loc[prior_nlv.isna() | (prior_nlv == 0), "return_daypl_prior_nlv"] = 0.0
    d["return_daypl_prior_nlv"] = d["return_daypl_prior_nlv"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Return basis 5: raw NLV pct change. Not recommended for MC because it includes deposits/withdrawals.
    d["return_raw_nlv"] = d["nlv"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Return basis 6: explicit return column, if available.
    d["return_column"] = d["return_col_value"].fillna(0.0)

    d["is_cashflow_day"] = d["cash_flow"].fillna(0.0).abs() > 1e-9
    d["is_after_cashflow_day"] = d["is_cashflow_day"].shift(1).fillna(False)
    d["is_weekday"] = pd.to_datetime(d["date"]).dt.weekday < 5
    d["source_kind"] = "portfolio_balance_history"
    d["source_file"] = filename

    notes.append(f"Loaded `{filename}` as portfolio balance history using `{date_col}` and `{balance_col}`.")
    if day_pnl_col:
        notes.append(f"Detected daily P/L column `{day_pnl_col}`.")
    if cashflow_col:
        notes.append(f"Detected deposits/withdrawals column `{cashflow_col}`.")
    if alltime_pct_col:
        notes.append(f"Detected broker cumulative return column `{alltime_pct_col}`.")
    notes.append("Default Monte Carlo basis uses adjusted equity returns and filters transfer days to avoid flat/distorted simulations.")

    return Scope(
        label="Portfolio Balance History",
        kind="portfolio",
        source_file=filename,
        data=d,
        notes=notes,
    )


def make_strategy_scopes(
    df: pd.DataFrame,
    filename: str,
    date_col: str,
    starting_balance: float,
    existing: Dict[str, Scope],
) -> Dict[str, Scope]:
    scopes: Dict[str, Scope] = {}
    notes: List[str] = []

    strategy_col = find_column(df, STRATEGY_CANDIDATES)
    balance_col = find_column(df, BALANCE_CANDIDATES)
    return_col = find_column(df, RETURN_CANDIDATES)
    pnl_col = find_column(df, PNL_CANDIDATES)

    w = df.copy()
    w["date"] = pd.to_datetime(w[date_col], errors="coerce").dt.tz_localize(None)
    w = w.dropna(subset=["date"]).copy()

    if strategy_col:
        w["_strategy"] = w[strategy_col].astype(str).replace("", "Unknown Strategy")
    else:
        w["_strategy"] = safe_stem(filename)

    if balance_col:
        w["_balance"] = to_numeric(w[balance_col])
    else:
        w["_balance"] = np.nan

    if return_col:
        w["_return_col"] = normalize_return_column(w[return_col], return_col)
    else:
        w["_return_col"] = np.nan

    if pnl_col:
        w["_pnl"] = to_numeric(w[pnl_col])
    else:
        w["_pnl"] = np.nan

    for strategy, group in w.groupby("_strategy"):
        group = group.sort_values("date").copy()
        daily = group.assign(_date=group["date"].dt.normalize()).groupby("_date", as_index=False).agg(
            balance=("_balance", "last"),
            return_column=("_return_col", "last"),
            pnl=("_pnl", "sum"),
        ).rename(columns={"_date": "date"}).sort_values("date").reset_index(drop=True)

        if daily["balance"].notna().any():
            daily["balance"] = daily["balance"].ffill()
            daily["return_strategy"] = daily["balance"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            method = f"balance column `{balance_col}`"
        elif daily["return_column"].notna().any():
            daily["return_strategy"] = daily["return_column"].fillna(0.0)
            daily["balance"] = starting_balance * (1 + daily["return_strategy"]).cumprod()
            method = f"return column `{return_col}`"
        elif daily["pnl"].notna().any():
            daily["pnl"] = daily["pnl"].fillna(0.0)
            daily["balance"] = starting_balance + daily["pnl"].cumsum()
            prior_balance = daily["balance"].shift(1)
            daily["return_strategy"] = daily["pnl"] / prior_balance
            daily.loc[prior_balance.isna() | (prior_balance == 0), "return_strategy"] = 0.0
            daily["return_strategy"] = daily["return_strategy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            method = f"P/L column `{pnl_col}` reconstructed from starting balance"
        else:
            continue

        daily["nlv"] = daily["balance"]
        daily["adjusted_equity"] = daily["balance"]
        daily["day_pl"] = daily["pnl"].fillna(0.0) if "pnl" in daily else 0.0
        daily["cash_flow"] = 0.0
        daily["cum_cash_flow"] = 0.0
        daily["is_cashflow_day"] = False
        daily["is_after_cashflow_day"] = False
        daily["is_weekday"] = pd.to_datetime(daily["date"]).dt.weekday < 5
        daily["source_kind"] = "strategy"
        daily["source_file"] = filename

        # Mirror return columns so the rest of the app can operate uniformly.
        for col in [
            "return_adjusted_equity",
            "return_daypl_prior_adjusted_equity",
            "return_broker_alltime_percent",
            "return_daypl_prior_nlv",
            "return_raw_nlv",
            "return_column",
        ]:
            daily[col] = daily["return_strategy"]

        label = dedupe(str(strategy).strip() or safe_stem(filename), {**existing, **scopes})
        scopes[label] = Scope(
            label=label,
            kind="strategy",
            source_file=filename,
            data=daily,
            notes=[f"Loaded `{filename}` as `{label}` using {method}."],
        )

    # Combined strategy scope when multiple strategies are present and P/L exists.
    if strategy_col and pnl_col and w["_pnl"].notna().any():
        combined = w.assign(_date=w["date"].dt.normalize()).groupby("_date", as_index=False).agg(
            pnl=("_pnl", "sum")
        ).rename(columns={"_date": "date"}).sort_values("date").reset_index(drop=True)

        combined["balance"] = starting_balance + combined["pnl"].fillna(0.0).cumsum()
        prior_balance = combined["balance"].shift(1)
        combined["return_strategy"] = combined["pnl"] / prior_balance
        combined.loc[prior_balance.isna() | (prior_balance == 0), "return_strategy"] = 0.0
        combined["return_strategy"] = combined["return_strategy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        combined["nlv"] = combined["balance"]
        combined["adjusted_equity"] = combined["balance"]
        combined["day_pl"] = combined["pnl"].fillna(0.0)
        combined["cash_flow"] = 0.0
        combined["cum_cash_flow"] = 0.0
        combined["is_cashflow_day"] = False
        combined["is_after_cashflow_day"] = False
        combined["is_weekday"] = pd.to_datetime(combined["date"]).dt.weekday < 5
        combined["source_kind"] = "strategy_combined"
        combined["source_file"] = filename

        for col in [
            "return_adjusted_equity",
            "return_daypl_prior_adjusted_equity",
            "return_broker_alltime_percent",
            "return_daypl_prior_nlv",
            "return_raw_nlv",
            "return_column",
        ]:
            combined[col] = combined["return_strategy"]

        label = dedupe("All Strategies - Combined P/L", {**existing, **scopes})
        scopes[label] = Scope(
            label=label,
            kind="strategy",
            source_file=filename,
            data=combined,
            notes=[f"Loaded combined strategy P/L from `{filename}` using `{pnl_col}`."],
        )

    return scopes


def parse_files(files, starting_balance: float) -> Dict[str, Scope]:
    scopes: Dict[str, Scope] = {}

    for uploaded in files:
        df = clean_columns(read_csv(uploaded))
        date_col = find_column(df, DATE_CANDIDATES)
        if not date_col:
            continue

        balance_col = find_column(df, BALANCE_CANDIDATES)
        has_nlv = balance_col and normalize_col_name(balance_col) in {
            "nlv", "net liq", "netliq", "net liquidation", "net liquidation value", "netliquidation"
        }
        has_cashflows = find_column(df, CASHFLOW_CANDIDATES) is not None
        has_day_pl = find_column(df, DAY_PNL_CANDIDATES) is not None

        # Broker portfolio files get their own special parser.
        if has_nlv and (has_cashflows or has_day_pl):
            scope = make_portfolio_scope(df, uploaded.name, date_col, balance_col)
            scopes[dedupe(scope.label, scopes)] = scope

        # Strategy scopes are still attempted if strategy-like columns exist.
        strategy_col = find_column(df, STRATEGY_CANDIDATES)
        pnl_col = find_column(df, PNL_CANDIDATES)
        return_col = find_column(df, RETURN_CANDIDATES)

        if strategy_col or (not has_nlv and (balance_col or pnl_col or return_col)):
            scopes.update(make_strategy_scopes(df, uploaded.name, date_col, starting_balance, scopes))

    return scopes


# ------------------------------------------------------------
# Analytics
# ------------------------------------------------------------

RETURN_BASIS_OPTIONS = {
    "Adjusted equity return — removes deposits/withdrawals": "return_adjusted_equity",
    "Day P/L ÷ prior adjusted equity": "return_daypl_prior_adjusted_equity",
    "Broker AllTime_PL_Percent daily change": "return_broker_alltime_percent",
    "Day P/L ÷ prior NLV — conservative total-account return": "return_daypl_prior_nlv",
    "Raw NLV % change — includes deposits/withdrawals, not recommended for MC": "return_raw_nlv",
    "Return column from file": "return_column",
}


def clean_returns(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_sitout(returns: pd.Series, dates: pd.Series, start_date: pd.Timestamp, months: int) -> Tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    adjusted = clean_returns(returns).copy()
    start = pd.Timestamp(start_date).normalize()
    end = start + pd.DateOffset(months=months)
    mask = (pd.to_datetime(dates) >= start) & (pd.to_datetime(dates) < end)
    adjusted.loc[mask.values] = 0.0
    return adjusted, start, end


def equity_curve(returns: pd.Series, starting_value: float) -> pd.Series:
    r = clean_returns(returns)
    return starting_value * (1 + r).cumprod()


def filtered_monte_carlo_returns(
    df: pd.DataFrame,
    return_col: str,
    sampling_mode: str,
    exclude_transfer_days: bool,
    min_abs_return: float = 0.000001,
) -> pd.Series:
    data = df.copy()
    r = clean_returns(data[return_col])
    mask = pd.Series(True, index=data.index)

    if sampling_mode == "Weekdays only":
        mask &= data["is_weekday"].fillna(True).astype(bool)
    elif sampling_mode == "Active trading days only":
        mask &= r.abs() >= min_abs_return

    if exclude_transfer_days and "is_cashflow_day" in data.columns:
        mask &= ~data["is_cashflow_day"].fillna(False).astype(bool)
        mask &= ~data["is_after_cashflow_day"].fillna(False).astype(bool)

    out = r[mask].dropna()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # Always remove the first zero observation from pct_change style series.
    if len(out) > 1 and abs(out.iloc[0]) < 1e-15:
        out = out.iloc[1:]

    return out


def monte_carlo_paths(
    returns: pd.Series,
    starting_value: float,
    years: int,
    simulations: int,
    sitout_months: int = 0,
    seed: int = 42,
) -> np.ndarray:
    r = clean_returns(returns).dropna()
    if len(r) < 5:
        raise ValueError("Not enough return observations for Monte Carlo analysis after filtering.")

    trading_days = int(252 * years)
    sitout_days = int(round(21 * sitout_months))

    rng = np.random.default_rng(seed)
    sampled = rng.choice(r.values, size=(simulations, trading_days), replace=True)

    if sitout_days > 0:
        sampled[:, : min(sitout_days, trading_days)] = 0.0

    return starting_value * np.cumprod(1 + sampled, axis=1)


def path_summary(paths: np.ndarray) -> Dict[str, float]:
    end = paths[:, -1]
    return {
        "p5": float(np.percentile(end, 5)),
        "p25": float(np.percentile(end, 25)),
        "median": float(np.percentile(end, 50)),
        "p75": float(np.percentile(end, 75)),
        "p95": float(np.percentile(end, 95)),
        "mean": float(np.mean(end)),
    }


def summarize_return_stream(r: pd.Series) -> Dict[str, float]:
    r = clean_returns(r).dropna()
    if r.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "annualized_mean": np.nan}
    return {
        "count": int(len(r)),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "annualized_mean": float((1 + r.mean()) ** 252 - 1),
    }


def money(x) -> str:
    if pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def pct(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.2%}"


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def historical_overlay_chart(dates, full_curve, sitout_curve, sitout_start, sitout_end) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=full_curve, mode="lines", name="Full participation"))
    fig.add_trace(go.Scatter(x=dates, y=sitout_curve, mode="lines", name="Sit-out counterfactual"))
    fig.add_vrect(
        x0=sitout_start,
        x1=sitout_end,
        fillcolor="gray",
        opacity=0.15,
        line_width=0,
        annotation_text="Sit-out period",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Historical Overlay: Full Participation vs. Sitting Out",
        xaxis_title="Date",
        yaxis_title="Modeled Account Value",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def projection_chart(paths_full, paths_sit, last_date, years, sitout_months) -> go.Figure:
    dates = pd.bdate_range(pd.Timestamp(last_date) + pd.offsets.BDay(1), periods=paths_full.shape[1])
    p_full = {
        "p5": np.percentile(paths_full, 5, axis=0),
        "p50": np.percentile(paths_full, 50, axis=0),
        "p95": np.percentile(paths_full, 95, axis=0),
    }
    p_sit = {
        "p50": np.percentile(paths_sit, 50, axis=0),
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=p_full["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=p_full["p5"], line=dict(width=0), fill="tonexty", name="Full 5th-95th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=p_full["p50"], mode="lines", name="Full participation median"))
    fig.add_trace(go.Scatter(x=dates, y=p_sit["p50"], mode="lines", name=f"Sit out first {sitout_months} months median"))
    fig.update_layout(
        title=f"{years}-Year Monte Carlo Projection",
        xaxis_title="Date",
        yaxis_title="Projected Account Value",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def distribution_chart(paths_full, paths_sit, years, sitout_months) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=paths_full[:, -1], histnorm="probability density", name="Full participation", opacity=0.6, nbinsx=60))
    fig.add_trace(go.Histogram(x=paths_sit[:, -1], histnorm="probability density", name=f"Sit out first {sitout_months} months", opacity=0.6, nbinsx=60))
    fig.update_layout(
        title=f"{years}-Year Ending Value Distribution",
        xaxis_title="Ending Account Value",
        yaxis_title="Probability Density",
        barmode="overlay",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("Inputs")

    files = st.file_uploader(
        "Upload portfolio or strategy CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    strategy_starting_balance = st.number_input(
        "Starting balance for P/L-only strategy files",
        min_value=1000.0,
        value=100000.0,
        step=5000.0,
        format="%.2f",
    )

    st.divider()
    st.subheader("Sit-out settings")
    sitout_months = st.slider("Sit-out months", 1, 12, 3, 1)

    st.divider()
    st.subheader("Monte Carlo settings")
    simulations = st.slider("Simulations", 250, 5000, 1000, 250)
    seed = st.number_input("Random seed", value=42, step=1)

if not files:
    st.info("Upload your portfolio balance-history CSV or one or more strategy CSV files.")
    st.stop()

scopes = parse_files(files, strategy_starting_balance)

if not scopes:
    st.error("No usable data was found. The file needs a date column and either NLV/balance, return, or P/L data.")
    st.stop()


# ------------------------------------------------------------
# Scope selection
# ------------------------------------------------------------

scope_labels = list(scopes.keys())
selected_label = st.sidebar.selectbox("Analysis scope", scope_labels)
scope = scopes[selected_label]
df = scope.data.copy().sort_values("date").reset_index(drop=True)

with st.expander("Import notes", expanded=False):
    for label, sc in scopes.items():
        st.markdown(f"**{label}**")
        for note in sc.notes:
            st.write(f"- {note}")

st.subheader(selected_label)

if scope.kind == "portfolio":
    available_basis = list(RETURN_BASIS_OPTIONS.keys())
    # Default to adjusted equity. This removes cash flows and avoids the flat result caused by Day_PL/prior raw NLV.
    default_basis_index = available_basis.index("Adjusted equity return — removes deposits/withdrawals")
else:
    available_basis = [
        "Adjusted equity return — removes deposits/withdrawals",
        "Return column from file",
        "Raw NLV % change — includes deposits/withdrawals, not recommended for MC",
    ]
    default_basis_index = 0

basis_label = st.sidebar.selectbox(
    "Return basis",
    available_basis,
    index=default_basis_index,
    help="For portfolio files, adjusted equity removes deposits/withdrawals. Day P/L ÷ prior NLV is conservative and can look flat when NLV contains idle capital.",
)
return_col = RETURN_BASIS_OPTIONS[basis_label]

if df[return_col].isna().all():
    st.warning(f"The selected return basis `{basis_label}` is unavailable for this file. Falling back to adjusted equity returns.")
    return_col = "return_adjusted_equity"
    basis_label = "Adjusted equity return — removes deposits/withdrawals"

sampling_mode = st.sidebar.selectbox(
    "Monte Carlo sampling days",
    ["Weekdays only", "Active trading days only", "All calendar days"],
    index=0,
    help="The prior projection was closer to a trading-day model. All calendar days includes weekends/zero-return days and can make the simulation look flat.",
)

exclude_transfer_days = st.sidebar.checkbox(
    "Exclude deposit/withdrawal days and following day",
    value=True if scope.kind == "portfolio" else False,
    help="Broker balance exports can show artificial P/L jumps around transfers. Excluding those days keeps the return stream cleaner.",
)

first_date = pd.to_datetime(df["date"]).min().normalize()
last_date = pd.to_datetime(df["date"]).max().normalize()
default_sitout_start = max(first_date, last_date - pd.DateOffset(months=sitout_months))

sitout_start_date = st.sidebar.date_input(
    "Historical sit-out start date",
    value=default_sitout_start.date(),
    min_value=first_date.date(),
    max_value=last_date.date(),
)

projection_starting_value_default = float(df["nlv"].dropna().iloc[-1]) if "nlv" in df else float(df["adjusted_equity"].dropna().iloc[-1])
projection_starting_value = st.sidebar.number_input(
    "Projection starting value",
    min_value=0.0,
    value=projection_starting_value_default,
    step=5000.0,
    format="%.2f",
)


# ------------------------------------------------------------
# Historical overlay
# ------------------------------------------------------------

returns = clean_returns(df[return_col])
historical_starting_value = float(df["nlv"].iloc[0]) if "nlv" in df else projection_starting_value
full_curve = equity_curve(returns, historical_starting_value)
sit_returns, sitout_start, sitout_end = apply_sitout(returns, df["date"], pd.Timestamp(sitout_start_date), sitout_months)
sit_curve = equity_curve(sit_returns, historical_starting_value)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Start", str(first_date.date()))
col2.metric("Data End", str(last_date.date()))
col3.metric("Historical Modeled Return", pct(full_curve.iloc[-1] / full_curve.iloc[0] - 1))
col4.metric("Current NLV / Balance", money(projection_starting_value_default))

if scope.kind == "portfolio":
    extra1, extra2, extra3 = st.columns(3)
    extra1.metric("Final NLV", money(df["nlv"].iloc[-1]))
    extra2.metric("Final Adjusted Equity", money(df["adjusted_equity"].iloc[-1]))
    if "alltime_pl_percent" in df and df["alltime_pl_percent"].notna().any():
        extra3.metric("Broker AllTime PL %", f"{df['alltime_pl_percent'].dropna().iloc[-1]:.2f}%")
    else:
        extra3.metric("Broker AllTime PL %", "—")

fig = historical_overlay_chart(df["date"], full_curve, sit_curve, sitout_start, sitout_end)
st.plotly_chart(fig, use_container_width=True)

gap = float(full_curve.iloc[-1] - sit_curve.iloc[-1])
gap_label = "Opportunity cost of sitting out" if gap > 0 else "Capital protected by sitting out" if gap < 0 else "No difference"

c1, c2, c3 = st.columns(3)
c1.metric("Full Participation Final", money(full_curve.iloc[-1]))
c2.metric("Sit-Out Final", money(sit_curve.iloc[-1]))
c3.metric(gap_label, money(abs(gap)))


# ------------------------------------------------------------
# Return diagnostics
# ------------------------------------------------------------

mc_returns = filtered_monte_carlo_returns(
    df,
    return_col=return_col,
    sampling_mode=sampling_mode,
    exclude_transfer_days=exclude_transfer_days,
)

stats = summarize_return_stream(mc_returns)

st.subheader("Monte Carlo Return Stream")

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Sample Days Used", f"{stats['count']:,}")
d2.metric("Average Return", pct(stats["mean"]))
d3.metric("Daily Volatility", pct(stats["std"]))
d4.metric("Annualized Mean", pct(stats["annualized_mean"]))
d5.metric("Worst Sample Day", pct(stats["min"]))

with st.expander("Why the earlier simulation looked flat", expanded=False):
    st.markdown(
        """
The flat result usually comes from two issues in portfolio balance-history exports:

1. **Using `Day_PL / prior NLV` can understate the strategy return** when NLV includes idle capital, deposits, or withdrawals.
2. **Sampling all calendar days includes weekends and zero-return days**, which dilutes the return stream in a 252-trading-day Monte Carlo model.

This build defaults to **adjusted equity returns**, **weekdays only**, and **excludes transfer days plus the following day** for portfolio exports. You can change those settings in the sidebar.
"""
    )

fig_ret = go.Figure()
fig_ret.add_trace(go.Histogram(x=mc_returns, nbinsx=60, name="Monte Carlo sample returns"))
fig_ret.update_layout(
    title="Filtered Return Distribution Used for Monte Carlo",
    xaxis_title="Daily Return",
    yaxis_title="Count",
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig_ret, use_container_width=True)


# ------------------------------------------------------------
# Projections
# ------------------------------------------------------------

st.subheader("Projection Overlays")

try:
    paths_1_full = monte_carlo_paths(
        mc_returns,
        starting_value=projection_starting_value,
        years=1,
        simulations=simulations,
        sitout_months=0,
        seed=int(seed),
    )
    paths_1_sit = monte_carlo_paths(
        mc_returns,
        starting_value=projection_starting_value,
        years=1,
        simulations=simulations,
        sitout_months=sitout_months,
        seed=int(seed),
    )
    paths_10_full = monte_carlo_paths(
        mc_returns,
        starting_value=projection_starting_value,
        years=10,
        simulations=simulations,
        sitout_months=0,
        seed=int(seed),
    )
    paths_10_sit = monte_carlo_paths(
        mc_returns,
        starting_value=projection_starting_value,
        years=10,
        simulations=simulations,
        sitout_months=sitout_months,
        seed=int(seed),
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.markdown("### 1-Year Projection")
st.plotly_chart(projection_chart(paths_1_full, paths_1_sit, last_date, 1, sitout_months), use_container_width=True)
st.plotly_chart(distribution_chart(paths_1_full, paths_1_sit, 1, sitout_months), use_container_width=True)

s1f = path_summary(paths_1_full)
s1s = path_summary(paths_1_sit)

p1, p2, p3, p4 = st.columns(4)
p1.metric("1Y Full Median", money(s1f["median"]))
p2.metric("1Y Sit-Out Median", money(s1s["median"]))
p3.metric("1Y Median Difference", money(abs(s1f["median"] - s1s["median"])))
p4.metric("1Y Full 5th Percentile", money(s1f["p5"]))

st.markdown("### 10-Year Projection")
st.plotly_chart(projection_chart(paths_10_full, paths_10_sit, last_date, 10, sitout_months), use_container_width=True)
st.plotly_chart(distribution_chart(paths_10_full, paths_10_sit, 10, sitout_months), use_container_width=True)

s10f = path_summary(paths_10_full)
s10s = path_summary(paths_10_sit)

q1, q2, q3, q4 = st.columns(4)
q1.metric("10Y Full Median", money(s10f["median"]))
q2.metric("10Y Sit-Out Median", money(s10s["median"]))
q3.metric("10Y Median Difference", money(abs(s10f["median"] - s10s["median"])))
q4.metric("10Y Full 5th Percentile", money(s10f["p5"]))


summary = pd.DataFrame(
    [
        {"Horizon": "1 Year", "Scenario": "Full Participation", **s1f},
        {"Horizon": "1 Year", "Scenario": f"Sit Out First {sitout_months} Months", **s1s},
        {"Horizon": "10 Years", "Scenario": "Full Participation", **s10f},
        {"Horizon": "10 Years", "Scenario": f"Sit Out First {sitout_months} Months", **s10s},
    ]
)

summary = summary.rename(
    columns={
        "p5": "5th Percentile",
        "p25": "25th Percentile",
        "median": "Median",
        "p75": "75th Percentile",
        "p95": "95th Percentile",
        "mean": "Mean",
    }
)

st.subheader("Projection Summary")
money_cols = ["5th Percentile", "25th Percentile", "Median", "75th Percentile", "95th Percentile", "Mean"]
st.dataframe(
    summary.style.format({col: "${:,.0f}" for col in money_cols}),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Monte Carlo projections are generated by resampling the selected historical daily return stream. "
    "They are not predictions. The results are highly sensitive to return basis, sample-day filtering, and whether transfer days are included."
)
