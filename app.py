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
# Portfolio balance history + individual strategy analysis
# Sit-out overlay + Monte Carlo projections
# ============================================================

st.set_page_config(
    page_title="ALGO Edge Performance History",
    page_icon="📈",
    layout="wide",
)

st.title("ALGO Edge Performance History")
st.caption(
    "Portfolio balance-history analysis, individual strategy analysis, sit-out overlays, and Monte Carlo projections."
)

st.markdown(
    """
This app supports both:

1. **Portfolio balance-history CSVs** with columns like `Date`, `NLV`, `Day_PL`, `Day_PL_Percent`, and `Deposits/Withdrawals`.
2. **Individual strategy CSVs** with a date column and either balance, return, or P/L columns. If a `Strategy` column exists, the app lets you analyze each strategy separately.

For sit-out testing, the model treats the sit-out period as cash: returns are set to **0%** during the selected window, then normal compounding resumes.
"""
)


# -----------------------------
# Column detection
# -----------------------------
DATE_CANDIDATES = [
    "date", "Date", "DATE", "datetime", "Datetime", "timestamp", "Timestamp",
    "time", "Time", "close_time", "Close Time", "Trade Date", "trade_date",
    "entry_date", "Entry Date", "exit_date", "Exit Date",
]

BALANCE_CANDIDATES = [
    "NLV", "nlv", "Net Liquidation Value", "net liquidation value",
    "NetLiq", "netliq", "net_liq", "Net Liq", "NetLiquidation", "Net Liquidation",
    "balance", "Balance", "BALANCE", "equity", "Equity",
    "account_value", "Account Value", "AccountValue",
    "value", "Value", "cumulative_balance", "Cumulative Balance",
    "ending_balance", "Ending Balance", "ending_value", "Ending Value",
]

DAY_PNL_CANDIDATES = [
    "Day_PL", "day_pl", "Day P/L", "Daily P/L", "Daily PL", "daily_pl",
    "Day PnL", "Daily PnL", "day_pnl", "daily_pnl",
]

CASHFLOW_CANDIDATES = [
    "Deposits/Withdrawals", "deposits/withdrawals",
    "Deposits Withdrawals", "deposits withdrawals",
    "Deposit/Withdrawal", "deposit/withdrawal",
    "Cash Flow", "cash_flow", "CashFlow", "Net Deposits", "net_deposits",
    "Deposits", "deposits", "Withdrawals", "withdrawals",
]

RETURN_CANDIDATES = [
    "return", "Return", "returns", "Returns",
    "daily_return", "Daily Return",
    "pct_return", "Pct Return", "% Return",
    "percent_return", "Percent Return",
    "Day_PL_Percent", "day_pl_percent",
    "Day P/L Percent", "Daily P/L Percent", "Daily PL Percent",
]

PNL_CANDIDATES = [
    "pnl", "PnL", "P/L", "p/l",
    "profit", "Profit", "daily_pnl", "Daily PnL",
    "net_profit", "Net Profit", "Net P/L",
    "Realized P/L", "realized_pnl", "Realized PnL",
    "Trade P/L", "Trade PL", "trade_pnl",
]

STRATEGY_CANDIDATES = [
    "strategy", "Strategy", "STRATEGY",
    "system", "System",
    "setup", "Setup",
    "algo", "Algo",
    "name", "Name",
]


@dataclass
class ParsedData:
    scopes: Dict[str, pd.DataFrame]
    notes: List[str]
    preview: pd.DataFrame


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    exact = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    normalized_candidates = [_normalize_col_name(c) for c in candidates]
    for col in df.columns:
        normalized_col = _normalize_col_name(col)
        for candidate in normalized_candidates:
            if candidate and candidate == normalized_col:
                return col

    # Fuzzy fallback for broker exports like "Account Balance ($)".
    for col in df.columns:
        normalized_col = _normalize_col_name(col)
        for candidate in normalized_candidates:
            if candidate and candidate in normalized_col:
                return col

    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.BytesIO(content), engine="python")


def _safe_file_stem(filename: str) -> str:
    stem = str(filename).rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("_", " ").replace("-", " ").strip() or "Uploaded File"


def _dedupe_label(label: str, existing: Dict[str, pd.DataFrame]) -> str:
    if label not in existing:
        return label

    i = 2
    while f"{label} ({i})" in existing:
        i += 1
    return f"{label} ({i})"


def _normalize_return_series(series: pd.Series) -> pd.Series:
    values = _to_numeric(series)
    median_abs = values.abs().median(skipna=True)

    # Return columns may come in as 0.45 or 45, or as 0.45 meaning 0.45%.
    # For common broker percent columns like Day_PL_Percent, values such as 0.45 mean 0.45%.
    # If the median absolute value is greater than 0.5, assume percent notation.
    # If the column contains names with "Percent", the caller handles conversion explicitly.
    if pd.notna(median_abs) and median_abs > 1:
        values = values / 100.0

    return values


def _build_curve_from_returns_and_cashflows(
    returns: pd.Series,
    cashflows: pd.Series,
    initial_balance: float,
    sitout_mask: Optional[pd.Series] = None,
) -> pd.Series:
    returns = pd.Series(returns).fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(float)
    cashflows = pd.Series(cashflows).fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(float)

    if sitout_mask is None:
        sitout_mask = pd.Series(False, index=returns.index)
    else:
        sitout_mask = pd.Series(sitout_mask, index=returns.index).fillna(False)

    balance = float(initial_balance)
    values: List[float] = []

    for i in range(len(returns)):
        r = 0.0 if bool(sitout_mask.iloc[i]) else float(returns.iloc[i])
        cf = float(cashflows.iloc[i])

        # First balance-history row is usually the starting NLV with zero return/cashflow.
        if i == 0 and abs(r) < 1e-15 and abs(cf) < 1e-15:
            values.append(balance)
            continue

        balance = balance + cf + (balance * r)
        values.append(balance)

    return pd.Series(values, index=returns.index)


def _build_scope_dataframe(
    daily: pd.DataFrame,
    initial_balance: float,
    scope_label: str,
    source_file: str,
    source_kind: str,
) -> pd.DataFrame:
    daily = daily.copy().sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["return"] = pd.to_numeric(daily["return"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["cash_flow"] = pd.to_numeric(daily.get("cash_flow", 0.0), errors="coerce").fillna(0.0)
    daily["balance"] = pd.to_numeric(daily["balance"], errors="coerce")

    daily["model_balance"] = _build_curve_from_returns_and_cashflows(
        daily["return"],
        daily["cash_flow"],
        initial_balance,
    )

    # For actual balance-history files, keep actual balance visible.
    # For return/P&L reconstructed files, actual and model balance are generally the same.
    daily["scope"] = scope_label
    daily["source_file"] = source_file
    daily["source_kind"] = source_kind
    daily["initial_balance"] = float(initial_balance)

    return daily


def _make_scope_from_balance_file(
    df: pd.DataFrame,
    filename: str,
    date_col: str,
    balance_col: str,
    starting_balance_for_pnl: float,
    strategy_col: Optional[str],
    notes: List[str],
) -> Dict[str, pd.DataFrame]:
    scopes: Dict[str, pd.DataFrame] = {}

    day_pnl_col = _find_column(df, DAY_PNL_CANDIDATES)
    cashflow_col = _find_column(df, CASHFLOW_CANDIDATES)
    return_col = _find_column(df, RETURN_CANDIDATES)

    working = df.copy()
    working["date"] = pd.to_datetime(working[date_col], errors="coerce").dt.tz_localize(None)
    working["balance_value"] = _to_numeric(working[balance_col])
    working["cash_flow_value"] = _to_numeric(working[cashflow_col]) if cashflow_col else 0.0
    working["day_pnl_value"] = _to_numeric(working[day_pnl_col]) if day_pnl_col else np.nan

    if return_col:
        return_values = _to_numeric(working[return_col])
        # Day_PL_Percent is expressed in percentage points: -0.95 means -0.95%, not -95%.
        if "percent" in _normalize_col_name(return_col) or "%" in str(return_col):
            return_values = return_values / 100.0
        else:
            return_values = _normalize_return_series(working[return_col])
        working["return_value_from_column"] = return_values
    else:
        working["return_value_from_column"] = np.nan

    working = working.dropna(subset=["date", "balance_value"]).copy()
    if working.empty:
        notes.append(f"Skipped `{filename}` because the balance column `{balance_col}` had no usable numeric values.")
        return scopes

    if strategy_col:
        group_iter = working.groupby(working[strategy_col].astype(str).fillna("Unknown Strategy"))
    else:
        # NLV-style files are portfolio account history files.
        balance_name = _normalize_col_name(balance_col)
        if balance_name in ["nlv", "net liq", "net liquidation", "net liquidation value", "netliquidation"]:
            label = "Portfolio Balance History"
        else:
            label = _safe_file_stem(filename)
        group_iter = [(label, working)]

    for group_name, group in group_iter:
        group = group.sort_values("date").copy()

        # Aggregate to one row per calendar date.
        agg = {
            "balance_value": "last",
            "cash_flow_value": "sum",
            "day_pnl_value": "sum",
            "return_value_from_column": "last",
        }

        daily = (
            group.assign(_daily_date=group["date"].dt.normalize())
            .groupby("_daily_date", as_index=False)
            .agg(agg)
            .rename(columns={"_daily_date": "date"})
        )

        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        prior_balance = daily["balance_value"].shift(1)

        if day_pnl_col and daily["day_pnl_value"].notna().any():
            # Best for balance-history exports because it removes deposit/withdrawal distortion.
            daily["return"] = daily["day_pnl_value"] / prior_balance
            daily.loc[prior_balance.isna() | (prior_balance == 0), "return"] = 0.0
            return_source = f"`{day_pnl_col}` / prior balance"
        elif return_col and daily["return_value_from_column"].notna().any():
            daily["return"] = daily["return_value_from_column"].fillna(0.0)
            return_source = f"`{return_col}`"
        else:
            daily["return"] = daily["balance_value"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return_source = f"percentage change in `{balance_col}`"

        daily["cash_flow"] = daily["cash_flow_value"].fillna(0.0)
        daily["balance"] = daily["balance_value"]

        initial_balance = float(daily["balance"].iloc[0])
        scope_label = str(group_name).strip() or _safe_file_stem(filename)
        scope_label = _dedupe_label(scope_label, scopes)

        scopes[scope_label] = _build_scope_dataframe(
            daily[["date", "balance", "return", "cash_flow"]],
            initial_balance=initial_balance,
            scope_label=scope_label,
            source_file=filename,
            source_kind="balance_history",
        )

        cf_note = f" and cash-flow column `{cashflow_col}`" if cashflow_col else ""
        notes.append(
            f"Loaded `{filename}` as `{scope_label}` using `{date_col}`, balance column `{balance_col}`, "
            f"returns from {return_source}{cf_note}."
        )

    return scopes


def _make_scope_from_return_file(
    df: pd.DataFrame,
    filename: str,
    date_col: str,
    return_col: str,
    starting_balance_for_pnl: float,
    strategy_col: Optional[str],
    notes: List[str],
) -> Dict[str, pd.DataFrame]:
    scopes: Dict[str, pd.DataFrame] = {}
    working = df.copy()
    working["date"] = pd.to_datetime(working[date_col], errors="coerce").dt.tz_localize(None)

    returns = _to_numeric(working[return_col])
    if "percent" in _normalize_col_name(return_col) or "%" in str(return_col):
        returns = returns / 100.0
    else:
        returns = _normalize_return_series(working[return_col])

    working["return_value"] = returns
    working = working.dropna(subset=["date", "return_value"]).copy()

    if working.empty:
        notes.append(f"Skipped `{filename}` because return column `{return_col}` had no usable numeric values.")
        return scopes

    if strategy_col:
        group_iter = working.groupby(working[strategy_col].astype(str).fillna("Unknown Strategy"))
    else:
        group_iter = [(_safe_file_stem(filename), working)]

    for group_name, group in group_iter:
        sorted_group = group.sort_values("date").assign(_daily_date=group["date"].dt.normalize())
        daily = (
            sorted_group
            .groupby("_daily_date", as_index=False)
            .agg({"return_value": "last"})
            .rename(columns={"_daily_date": "date"})
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["return"] = daily["return_value"].fillna(0.0)
        daily["cash_flow"] = 0.0

        balances = _build_curve_from_returns_and_cashflows(
            daily["return"],
            daily["cash_flow"],
            starting_balance_for_pnl,
        )
        daily["balance"] = balances

        scope_label = _dedupe_label(str(group_name).strip() or _safe_file_stem(filename), scopes)
        scopes[scope_label] = _build_scope_dataframe(
            daily[["date", "balance", "return", "cash_flow"]],
            initial_balance=starting_balance_for_pnl,
            scope_label=scope_label,
            source_file=filename,
            source_kind="return_history",
        )

        notes.append(
            f"Loaded `{filename}` as `{scope_label}` using `{date_col}` and return column `{return_col}`."
        )

    return scopes


def _make_scope_from_pnl_file(
    df: pd.DataFrame,
    filename: str,
    date_col: str,
    pnl_col: str,
    starting_balance_for_pnl: float,
    strategy_col: Optional[str],
    notes: List[str],
) -> Dict[str, pd.DataFrame]:
    scopes: Dict[str, pd.DataFrame] = {}

    working = df.copy()
    working["date"] = pd.to_datetime(working[date_col], errors="coerce").dt.tz_localize(None)
    working["pnl_value"] = _to_numeric(working[pnl_col])
    working = working.dropna(subset=["date", "pnl_value"]).copy()

    if working.empty:
        notes.append(f"Skipped `{filename}` because P/L column `{pnl_col}` had no usable numeric values.")
        return scopes

    if strategy_col:
        group_iter = working.groupby(working[strategy_col].astype(str).fillna("Unknown Strategy"))
    else:
        group_iter = [(_safe_file_stem(filename), working)]

    # Individual strategy/file scopes.
    for group_name, group in group_iter:
        sorted_group = group.sort_values("date").assign(_daily_date=group["date"].dt.normalize())
        daily = (
            sorted_group
            .groupby("_daily_date", as_index=False)
            .agg({"pnl_value": "sum"})
            .rename(columns={"_daily_date": "date"})
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["cash_flow"] = 0.0

        balance = float(starting_balance_for_pnl)
        balances = []
        returns = []
        for pnl in daily["pnl_value"].fillna(0.0):
            r = pnl / balance if balance else 0.0
            balance = balance + pnl
            returns.append(r)
            balances.append(balance)

        daily["return"] = returns
        daily["balance"] = balances

        scope_label = _dedupe_label(str(group_name).strip() or _safe_file_stem(filename), scopes)
        scopes[scope_label] = _build_scope_dataframe(
            daily[["date", "balance", "return", "cash_flow"]],
            initial_balance=starting_balance_for_pnl,
            scope_label=scope_label,
            source_file=filename,
            source_kind="pnl_history",
        )

        notes.append(
            f"Loaded `{filename}` as `{scope_label}` using `{date_col}` and P/L column `{pnl_col}`."
        )

    # If a Strategy column exists, also create a combined P/L scope for that file.
    if strategy_col:
        sorted_working = working.sort_values("date").assign(_daily_date=working["date"].dt.normalize())
        daily_all = (
            sorted_working
            .groupby("_daily_date", as_index=False)
            .agg({"pnl_value": "sum"})
            .rename(columns={"_daily_date": "date"})
        )
        daily_all["date"] = pd.to_datetime(daily_all["date"])
        daily_all = daily_all.sort_values("date").reset_index(drop=True)
        daily_all["cash_flow"] = 0.0

        balance = float(starting_balance_for_pnl)
        balances = []
        returns = []
        for pnl in daily_all["pnl_value"].fillna(0.0):
            r = pnl / balance if balance else 0.0
            balance = balance + pnl
            returns.append(r)
            balances.append(balance)

        daily_all["return"] = returns
        daily_all["balance"] = balances

        combined_label = _dedupe_label("All Strategies - Combined P/L", scopes)
        scopes[combined_label] = _build_scope_dataframe(
            daily_all[["date", "balance", "return", "cash_flow"]],
            initial_balance=starting_balance_for_pnl,
            scope_label=combined_label,
            source_file=filename,
            source_kind="combined_pnl_history",
        )

    return scopes


def parse_uploaded_files(files, starting_balance_for_pnl: float) -> ParsedData:
    all_scopes: Dict[str, pd.DataFrame] = {}
    notes: List[str] = []
    previews: List[pd.DataFrame] = []

    for file in files:
        try:
            df = _clean_columns(_read_uploaded_csv(file))
        except Exception as exc:
            notes.append(f"Skipped `{file.name}` because it could not be read as a CSV: {exc}")
            continue

        df["__source_file"] = file.name
        previews.append(df.head(10).copy())

        date_col = _find_column(df, DATE_CANDIDATES)
        if date_col is None:
            notes.append(f"Skipped `{file.name}` because no date column was detected.")
            continue

        strategy_col = _find_column(df, STRATEGY_CANDIDATES)
        balance_col = _find_column(df, BALANCE_CANDIDATES)
        return_col = _find_column(df, RETURN_CANDIDATES)
        pnl_col = _find_column(df, PNL_CANDIDATES)

        # Priority matters:
        # 1. Balance-history files with NLV/balance should be treated as balance files.
        # 2. Return files.
        # 3. P/L files.
        file_scopes: Dict[str, pd.DataFrame] = {}

        if balance_col is not None:
            file_scopes = _make_scope_from_balance_file(
                df=df,
                filename=file.name,
                date_col=date_col,
                balance_col=balance_col,
                starting_balance_for_pnl=starting_balance_for_pnl,
                strategy_col=strategy_col,
                notes=notes,
            )
        elif return_col is not None:
            file_scopes = _make_scope_from_return_file(
                df=df,
                filename=file.name,
                date_col=date_col,
                return_col=return_col,
                starting_balance_for_pnl=starting_balance_for_pnl,
                strategy_col=strategy_col,
                notes=notes,
            )
        elif pnl_col is not None:
            file_scopes = _make_scope_from_pnl_file(
                df=df,
                filename=file.name,
                date_col=date_col,
                pnl_col=pnl_col,
                starting_balance_for_pnl=starting_balance_for_pnl,
                strategy_col=strategy_col,
                notes=notes,
            )
        else:
            notes.append(
                f"Skipped `{file.name}` because no balance, return, or P/L column was detected."
            )

        for label, scope_df in file_scopes.items():
            safe_label = _dedupe_label(label, all_scopes)
            if safe_label != label:
                scope_df = scope_df.copy()
                scope_df["scope"] = safe_label
            all_scopes[safe_label] = scope_df

    preview = pd.concat(previews, ignore_index=True) if previews else pd.DataFrame()
    return ParsedData(scopes=all_scopes, notes=notes, preview=preview)


# -----------------------------
# Modeling / chart helpers
# -----------------------------
def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2%}"


def apply_sitout_curve(
    daily: pd.DataFrame,
    sitout_start: pd.Timestamp,
    sitout_months: int,
) -> Tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    sitout_start = pd.Timestamp(sitout_start).normalize()
    sitout_end = sitout_start + pd.DateOffset(months=sitout_months)

    dates = pd.to_datetime(daily["date"])
    mask = (dates >= sitout_start) & (dates < sitout_end)

    curve = _build_curve_from_returns_and_cashflows(
        daily["return"],
        daily["cash_flow"],
        float(daily["initial_balance"].iloc[0]),
        sitout_mask=mask,
    )

    return curve, sitout_start, sitout_end


def describe_gap(full_final: float, sitout_final: float) -> Tuple[str, float, float]:
    gap = full_final - sitout_final
    pct_gap = gap / full_final if full_final else np.nan

    if gap > 0:
        label = "Opportunity cost of sitting out"
    elif gap < 0:
        label = "Capital protected by sitting out"
    else:
        label = "No difference"

    return label, gap, pct_gap


def make_historical_overlay_chart(
    daily: pd.DataFrame,
    full_curve: pd.Series,
    sitout_curve: pd.Series,
    sitout_start: pd.Timestamp,
    sitout_end: pd.Timestamp,
    selected_scope: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=full_curve,
            mode="lines",
            name="Full Participation",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=sitout_curve,
            mode="lines",
            name="Sit Out",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    if "balance" in daily.columns and daily["source_kind"].iloc[0] == "balance_history":
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["balance"],
                mode="lines",
                name="Actual Reported Balance",
                line=dict(dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            )
        )

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
        title=f"Historical Overlay: {selected_scope}",
        xaxis_title="Date",
        yaxis_title="Account / Strategy Value",
        hovermode="x unified",
        legend_title_text="Scenario",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def monte_carlo_paths(
    historical_returns: pd.Series,
    starting_balance: float,
    years: int,
    simulations: int,
    sitout_months: int = 0,
    seed: int = 42,
) -> np.ndarray:
    clean_returns = (
        pd.Series(historical_returns)
        .dropna()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    if clean_returns.empty:
        raise ValueError("No valid historical returns are available for projection.")

    trading_days = int(252 * years)
    sitout_days = int(round(21 * sitout_months))

    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean_returns.values, size=(simulations, trading_days), replace=True)

    if sitout_days > 0:
        sampled[:, : min(sitout_days, trading_days)] = 0.0

    return starting_balance * np.cumprod(1 + sampled, axis=1)


def summarize_paths(paths: np.ndarray) -> pd.DataFrame:
    percentiles = [5, 25, 50, 75, 95]
    values = np.percentile(paths, percentiles, axis=0)
    return pd.DataFrame({f"p{p}": values[i] for i, p in enumerate(percentiles)})


def final_stats(paths: np.ndarray) -> Dict[str, float]:
    ending = paths[:, -1]
    return {
        "p5": float(np.percentile(ending, 5)),
        "p25": float(np.percentile(ending, 25)),
        "median": float(np.percentile(ending, 50)),
        "p75": float(np.percentile(ending, 75)),
        "p95": float(np.percentile(ending, 95)),
        "mean": float(np.mean(ending)),
    }


def make_projection_chart(
    full_paths: np.ndarray,
    sitout_paths: np.ndarray,
    start_date: pd.Timestamp,
    years: int,
    sitout_months: int,
    selected_scope: str,
) -> go.Figure:
    dates = pd.bdate_range(start=start_date + pd.offsets.BDay(1), periods=full_paths.shape[1])

    full = summarize_paths(full_paths)
    sitout = summarize_paths(sitout_paths)
    full["date"] = dates
    sitout["date"] = dates

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=full["date"],
            y=full["p95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="Full p95",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=full["date"],
            y=full["p5"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            opacity=0.15,
            name="Full 5th-95th Percentile",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=full["date"],
            y=full["p50"],
            mode="lines",
            name="Full Participation Median",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sitout["date"],
            y=sitout["p50"],
            mode="lines",
            name=f"Sit Out First {sitout_months} Months Median",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{years}-Year Projection Overlay: {selected_scope}",
        xaxis_title="Date",
        yaxis_title="Projected Value",
        hovermode="x unified",
        legend_title_text="Scenario",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def make_distribution_chart(
    full_paths: np.ndarray,
    sitout_paths: np.ndarray,
    years: int,
    sitout_months: int,
    selected_scope: str,
) -> go.Figure:
    full_final = full_paths[:, -1]
    sitout_final = sitout_paths[:, -1]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=full_final,
            histnorm="probability density",
            name="Full Participation",
            opacity=0.6,
            nbinsx=60,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=sitout_final,
            histnorm="probability density",
            name=f"Sit Out First {sitout_months} Months",
            opacity=0.6,
            nbinsx=60,
        )
    )

    fig.update_layout(
        title=f"{years}-Year Ending Value Distribution: {selected_scope}",
        xaxis_title="Ending Value",
        yaxis_title="Probability Density",
        barmode="overlay",
        legend_title_text="Scenario",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    uploaded_files = st.file_uploader(
        "Upload portfolio or strategy CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Portfolio balance-history format is supported: Date, NLV, Day_PL, Day_PL_Percent, Deposits/Withdrawals. "
            "Strategy files are also supported if they contain Date plus balance, return, or P/L columns."
        ),
    )

    starting_balance_for_pnl = st.number_input(
        "Starting balance for P/L-only strategy files",
        min_value=1.0,
        value=100000.0,
        step=5000.0,
        format="%.2f",
        help="Used only when a file has P/L but no balance column.",
    )

    st.divider()
    st.subheader("Sit-out overlay")

    sitout_months = st.slider(
        "Sit-out months",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
    )

    st.divider()
    st.subheader("Monte Carlo")

    simulations = st.slider(
        "Simulations",
        min_value=250,
        max_value=5000,
        value=1000,
        step=250,
    )

    exclude_zero_returns = st.checkbox(
        "Exclude 0% return days from Monte Carlo",
        value=True,
        help=(
            "Useful for balance-history files that include weekends/holidays as 0% days. "
            "Turn off if a no-trade day should count as a true 0% strategy day."
        ),
    )

    random_seed = st.number_input(
        "Random seed",
        value=42,
        step=1,
    )


if not uploaded_files:
    st.info(
        "Upload a portfolio balance-history CSV or one/more strategy CSV files to generate the analysis."
    )
    st.stop()


parsed = parse_uploaded_files(
    uploaded_files,
    starting_balance_for_pnl=starting_balance_for_pnl,
)

if parsed.notes:
    with st.expander("Import notes", expanded=False):
        for note in parsed.notes:
            st.write(f"- {note}")

if not parsed.scopes:
    st.error(
        "No usable data series were found. The app needs a date column plus one of: NLV/balance, return, or P/L."
    )
    if not parsed.preview.empty:
        st.write("Preview of uploaded data:")
        st.dataframe(parsed.preview.head(25), use_container_width=True)
    st.stop()


scope_options = list(parsed.scopes.keys())
default_index = 0
for i, option in enumerate(scope_options):
    if "portfolio" in option.lower():
        default_index = i
        break

with st.sidebar:
    selected_scope = st.selectbox(
        "Analysis scope",
        options=scope_options,
        index=default_index,
        help="Choose the portfolio balance history or an individual strategy.",
    )


daily = parsed.scopes[selected_scope].copy().sort_values("date").reset_index(drop=True)

if daily.empty or len(daily) < 3:
    st.error("The selected scope does not have enough rows for analysis.")
    st.stop()


first_date = pd.to_datetime(daily["date"]).min().normalize()
last_date = pd.to_datetime(daily["date"]).max().normalize()
default_sitout_start = max(first_date, last_date - pd.DateOffset(months=sitout_months))

with st.sidebar:
    sitout_start_date = st.date_input(
        "Sit-out start date",
        value=default_sitout_start.date(),
        min_value=first_date.date(),
        max_value=last_date.date(),
    )


# -----------------------------
# Historical overlay
# -----------------------------
sitout_curve, sitout_start, sitout_end = apply_sitout_curve(
    daily=daily,
    sitout_start=pd.Timestamp(sitout_start_date),
    sitout_months=sitout_months,
)

full_curve = daily["model_balance"].astype(float)
full_final = float(full_curve.iloc[-1])
sitout_final = float(sitout_curve.iloc[-1])

gap_label, gap_value, gap_pct = describe_gap(full_final, sitout_final)

st.subheader(f"Selected Analysis Scope: {selected_scope}")

source_kind = str(daily["source_kind"].iloc[0]).replace("_", " ").title()
st.caption(
    f"Source type: {source_kind} | Source file: {daily['source_file'].iloc[0]} | "
    f"Date range: {first_date.date()} to {last_date.date()}"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Starting Value", money(float(daily["initial_balance"].iloc[0])))
col2.metric("Full Participation Final", money(full_final))
col3.metric("Total Return", pct(full_final / float(daily["initial_balance"].iloc[0]) - 1))
col4.metric("Observations", f"{len(daily):,}")

fig_hist = make_historical_overlay_chart(
    daily=daily,
    full_curve=full_curve,
    sitout_curve=sitout_curve,
    sitout_start=sitout_start,
    sitout_end=sitout_end,
    selected_scope=selected_scope,
)

st.plotly_chart(fig_hist, use_container_width=True)


st.subheader("Sit-Out Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Full Participation Final Value", money(full_final))
col2.metric("Sit-Out Final Value", money(sitout_final))
col3.metric(gap_label, money(abs(gap_value)), pct(abs(gap_pct)))

if gap_value > 0:
    st.success(
        "In this selected window, sitting out reduced ending value versus full participation. "
        "That is the opportunity cost of missing the positive-drift return stream."
    )
elif gap_value < 0:
    st.warning(
        "In this selected window, sitting out improved ending value versus full participation. "
        "That means the avoided losses were larger than the missed gains."
    )
else:
    st.info("In this selected window, the sit-out and full-participation paths ended at the same value.")

with st.expander("How the sit-out overlay is calculated", expanded=False):
    st.markdown(
        f"""
- Returns from **{sitout_start.date()} through {(sitout_end - pd.Timedelta(days=1)).date()}** are set to **0%**.
- Cash flows, such as deposits or withdrawals, are preserved.
- After the sit-out window, the same historical return stream resumes.
- For portfolio balance-history files, returns are based on `Day_PL / prior NLV` when available. This avoids treating deposits as trading gains.
"""
    )


# -----------------------------
# Return diagnostics
# -----------------------------
st.subheader("Return Diagnostics")

returns_for_diagnostics = daily["return"].replace([np.inf, -np.inf], np.nan).dropna()

if exclude_zero_returns:
    returns_for_mc = returns_for_diagnostics[returns_for_diagnostics.abs() > 1e-12]
else:
    returns_for_mc = returns_for_diagnostics

returns_for_mc = returns_for_mc.dropna()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Daily Return", pct(returns_for_diagnostics.mean()))
col2.metric("Daily Volatility", pct(returns_for_diagnostics.std()))
col3.metric("Best Day", pct(returns_for_diagnostics.max()))
col4.metric("Worst Day", pct(returns_for_diagnostics.min()))

fig_ret = go.Figure()
fig_ret.add_trace(
    go.Histogram(
        x=returns_for_diagnostics,
        nbinsx=60,
        histnorm="probability density",
        name="Daily Returns",
    )
)
fig_ret.update_layout(
    title=f"Historical Daily Return Distribution: {selected_scope}",
    xaxis_title="Daily Return",
    yaxis_title="Probability Density",
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig_ret, use_container_width=True)

if exclude_zero_returns:
    st.caption(
        f"Monte Carlo will sample {len(returns_for_mc):,} non-zero return days out of {len(returns_for_diagnostics):,} total observations."
    )
else:
    st.caption(
        f"Monte Carlo will sample all {len(returns_for_mc):,} return observations, including 0% days."
    )

if len(returns_for_mc) < 5:
    st.error(
        "Not enough return observations are available for Monte Carlo. "
        "Try turning off 'Exclude 0% return days' or uploading a longer history."
    )
    st.stop()


# -----------------------------
# Monte Carlo projections
# -----------------------------
st.subheader("Projection Overlays")

projection_start_balance = float(full_curve.iloc[-1])

st.markdown(
    f"The projection compares normal compounding against sitting in cash for the first {sitout_months} months, then resuming the same sampled return profile."
)

try:
    paths_1yr_full = monte_carlo_paths(
        returns_for_mc,
        starting_balance=projection_start_balance,
        years=1,
        simulations=simulations,
        sitout_months=0,
        seed=int(random_seed),
    )

    paths_1yr_sitout = monte_carlo_paths(
        returns_for_mc,
        starting_balance=projection_start_balance,
        years=1,
        simulations=simulations,
        sitout_months=sitout_months,
        seed=int(random_seed),
    )

    paths_10yr_full = monte_carlo_paths(
        returns_for_mc,
        starting_balance=projection_start_balance,
        years=10,
        simulations=simulations,
        sitout_months=0,
        seed=int(random_seed),
    )

    paths_10yr_sitout = monte_carlo_paths(
        returns_for_mc,
        starting_balance=projection_start_balance,
        years=10,
        simulations=simulations,
        sitout_months=sitout_months,
        seed=int(random_seed),
    )

except ValueError as exc:
    st.error(str(exc))
    st.stop()


# 1-year projection
st.markdown("### 1-Year Projection")

fig_1yr = make_projection_chart(
    paths_1yr_full,
    paths_1yr_sitout,
    last_date,
    years=1,
    sitout_months=sitout_months,
    selected_scope=selected_scope,
)
st.plotly_chart(fig_1yr, use_container_width=True)

stats_1_full = final_stats(paths_1yr_full)
stats_1_sit = final_stats(paths_1yr_sitout)
median_gap_1 = stats_1_full["median"] - stats_1_sit["median"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Full Median", money(stats_1_full["median"]))
c2.metric("Sit-Out Median", money(stats_1_sit["median"]))
c3.metric("Median Difference", money(abs(median_gap_1)))
c4.metric("Full 5th Percentile", money(stats_1_full["p5"]))

fig_1yr_dist = make_distribution_chart(
    paths_1yr_full,
    paths_1yr_sitout,
    years=1,
    sitout_months=sitout_months,
    selected_scope=selected_scope,
)
st.plotly_chart(fig_1yr_dist, use_container_width=True)


# 10-year projection
st.markdown("### 10-Year Projection")

fig_10yr = make_projection_chart(
    paths_10yr_full,
    paths_10yr_sitout,
    last_date,
    years=10,
    sitout_months=sitout_months,
    selected_scope=selected_scope,
)
st.plotly_chart(fig_10yr, use_container_width=True)

stats_10_full = final_stats(paths_10yr_full)
stats_10_sit = final_stats(paths_10yr_sitout)
median_gap_10 = stats_10_full["median"] - stats_10_sit["median"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Full Median", money(stats_10_full["median"]))
c2.metric("Sit-Out Median", money(stats_10_sit["median"]))
c3.metric("Median Difference", money(abs(median_gap_10)))
c4.metric("Full 5th Percentile", money(stats_10_full["p5"]))

fig_10yr_dist = make_distribution_chart(
    paths_10yr_full,
    paths_10yr_sitout,
    years=10,
    sitout_months=sitout_months,
    selected_scope=selected_scope,
)
st.plotly_chart(fig_10yr_dist, use_container_width=True)


# -----------------------------
# Summary table
# -----------------------------
st.subheader("Projection Summary")

summary_df = pd.DataFrame(
    [
        {
            "Horizon": "1 Year",
            "Scenario": "Full Participation",
            "5th Percentile": stats_1_full["p5"],
            "25th Percentile": stats_1_full["p25"],
            "Median": stats_1_full["median"],
            "75th Percentile": stats_1_full["p75"],
            "95th Percentile": stats_1_full["p95"],
        },
        {
            "Horizon": "1 Year",
            "Scenario": f"Sit Out First {sitout_months} Months",
            "5th Percentile": stats_1_sit["p5"],
            "25th Percentile": stats_1_sit["p25"],
            "Median": stats_1_sit["median"],
            "75th Percentile": stats_1_sit["p75"],
            "95th Percentile": stats_1_sit["p95"],
        },
        {
            "Horizon": "10 Years",
            "Scenario": "Full Participation",
            "5th Percentile": stats_10_full["p5"],
            "25th Percentile": stats_10_full["p25"],
            "Median": stats_10_full["median"],
            "75th Percentile": stats_10_full["p75"],
            "95th Percentile": stats_10_full["p95"],
        },
        {
            "Horizon": "10 Years",
            "Scenario": f"Sit Out First {sitout_months} Months",
            "5th Percentile": stats_10_sit["p5"],
            "25th Percentile": stats_10_sit["p25"],
            "Median": stats_10_sit["median"],
            "75th Percentile": stats_10_sit["p75"],
            "95th Percentile": stats_10_sit["p95"],
        },
    ]
)

currency_cols = [
    "5th Percentile",
    "25th Percentile",
    "Median",
    "75th Percentile",
    "95th Percentile",
]

st.dataframe(
    summary_df.style.format({col: "${:,.0f}" for col in currency_cols}),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Parsed data preview", expanded=False):
    st.write("Selected scope data:")
    st.dataframe(daily.head(50), use_container_width=True)

st.caption(
    "Monte Carlo projections are based on random resampling of the selected historical return stream. "
    "They are not predictions; they show a distribution of possible outcomes if the historical return/volatility profile persists."
)
