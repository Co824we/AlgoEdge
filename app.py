import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# ALGO Edge Monte Carlo
# Actual-return Monte Carlo analysis
# ============================================================

st.set_page_config(
    page_title="ALGO Edge Monte Carlo",
    page_icon="📈",
    layout="wide",
)

st.title("ALGO Edge Monte Carlo")
st.caption("Monte Carlo projections based on actual realized returns.")

st.markdown(
    """
Upload a portfolio balance-history file or individual strategy files. The app builds a historical return stream,
then resamples those **actual realized returns** to generate 1-year and 10-year Monte Carlo projections.
"""
)


# -----------------------------
# Column detection
# -----------------------------
DATE_CANDIDATES = [
    "Date", "date", "DATE", "Datetime", "datetime", "Timestamp", "timestamp",
    "Trade Date", "trade_date", "Time", "time", "close_time", "Close Time",
]

BALANCE_CANDIDATES = [
    "NLV", "Net Liq", "Net Liquidation", "NetLiquidation", "balance", "Balance",
    "equity", "Equity", "account_value", "Account Value", "value", "Value",
    "ending_balance", "Ending Balance", "cumulative_balance", "Cumulative Balance",
]

RETURN_CANDIDATES = [
    "Day_PL_Percent", "Day P/L Percent", "Day_PL_%", "return", "Return", "returns", "Returns",
    "daily_return", "Daily Return", "pct_return", "Pct Return", "% Return", "percent_return",
    "Percent Return",
]

PNL_CANDIDATES = [
    "Day_PL", "Day P/L", "P/L", "p/l", "pnl", "PnL", "profit", "Profit",
    "daily_pnl", "Daily PnL", "net_profit", "Net Profit", "Net P/L", "realized_pnl",
    "Realized P/L",
]

DEPOSIT_CANDIDATES = [
    "Deposits/Withdrawals", "Deposits", "Withdrawals", "Deposit", "Withdrawal",
    "Cash Flow", "cash_flow", "Net Deposits", "net_deposits",
]

STRATEGY_CANDIDATES = [
    "Strategy", "strategy", "STRATEGY", "System", "system", "Name", "name",
]


@dataclass
class ScopeData:
    name: str
    source: str
    daily: pd.DataFrame
    notes: List[str]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    exact = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for col in df.columns:
        normalized = col.lower().replace("_", " ").replace("-", " ").strip()
        for candidate in candidates:
            c = candidate.lower().replace("_", " ").replace("-", " ").strip()
            if c == normalized or c in normalized:
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
        .str.strip(),
        errors="coerce",
    )


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.BytesIO(content), engine="python")


def _normalize_return(series: pd.Series) -> pd.Series:
    returns = _to_numeric(series)
    # If the file stores 1.25 for 1.25%, convert to decimal.
    median_abs = returns.abs().median(skipna=True)
    if pd.notna(median_abs) and median_abs > 1:
        returns = returns / 100.0
    return returns


def _daily_from_portfolio_balance_history(df: pd.DataFrame, file_name: str) -> Optional[ScopeData]:
    """Recognize broker-style portfolio export: Date, NLV, Day_PL, Day_PL_Percent, deposits/withdrawals."""
    date_col = _find_column(df, DATE_CANDIDATES)
    balance_col = _find_column(df, BALANCE_CANDIDATES)
    pnl_col = _find_column(df, PNL_CANDIDATES)
    ret_col = _find_column(df, RETURN_CANDIDATES)
    dep_col = _find_column(df, DEPOSIT_CANDIDATES)

    if date_col is None or balance_col is None:
        return None

    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.tz_localize(None)
    work["balance"] = _to_numeric(work[balance_col])
    work = work.dropna(subset=["date", "balance"]).sort_values("date")
    if work.empty:
        return None

    notes = [f"Loaded `{file_name}` as portfolio balance history using `{date_col}` and `{balance_col}`."]

    if pnl_col is not None:
        work["pnl"] = _to_numeric(work[pnl_col])
        # Actual realized trading return: day P/L divided by prior account value.
        work["return"] = work["pnl"] / work["balance"].shift(1)
        notes.append(f"Actual returns calculated from `{pnl_col}` divided by prior `{balance_col}`.")
    elif ret_col is not None:
        work["return"] = _normalize_return(work[ret_col])
        notes.append(f"Actual returns taken from `{ret_col}`.")
    else:
        work["return"] = work["balance"].pct_change()
        notes.append("Actual returns calculated from daily balance percentage change.")

    if dep_col is not None:
        work["deposit_withdrawal"] = _to_numeric(work[dep_col]).fillna(0.0)
        notes.append(f"Detected `{dep_col}`. Return stream still uses trading P/L when available so cash flows are not counted as trading returns.")
    else:
        work["deposit_withdrawal"] = 0.0

    daily = work[["date", "balance", "return", "deposit_withdrawal"]].copy()
    daily["return"] = daily["return"].replace([np.inf, -np.inf], np.nan)
    daily = daily.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return ScopeData(name="Portfolio — Actual Returns", source=file_name, daily=daily, notes=notes)


def _daily_from_generic_file(df: pd.DataFrame, file_name: str, starting_balance_for_pnl: float) -> List[ScopeData]:
    scopes: List[ScopeData] = []
    date_col = _find_column(df, DATE_CANDIDATES)
    balance_col = _find_column(df, BALANCE_CANDIDATES)
    ret_col = _find_column(df, RETURN_CANDIDATES)
    pnl_col = _find_column(df, PNL_CANDIDATES)
    strategy_col = _find_column(df, STRATEGY_CANDIDATES)

    if date_col is None:
        return scopes

    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.tz_localize(None)
    work = work.dropna(subset=["date"]).sort_values("date")
    if work.empty:
        return scopes

    if strategy_col is not None:
        groups = list(work.groupby(work[strategy_col].astype(str)))
    else:
        groups = [(file_name.rsplit(".", 1)[0], work)]

    for strategy_name, group in groups:
        g = group.copy().sort_values("date")
        notes = [f"Loaded `{file_name}` scope `{strategy_name}` using date column `{date_col}`."]

        if balance_col is not None:
            g["balance"] = _to_numeric(g[balance_col])
            g = g.dropna(subset=["balance"])
            if g.empty:
                continue
            g["return"] = g["balance"].pct_change()
            notes.append(f"Actual returns calculated from balance column `{balance_col}`.")

        elif ret_col is not None:
            g["return"] = _normalize_return(g[ret_col])
            g = g.dropna(subset=["return"])
            if g.empty:
                continue
            g["balance"] = starting_balance_for_pnl * (1 + g["return"].fillna(0.0)).cumprod()
            notes.append(f"Actual returns taken from return column `{ret_col}`.")

        elif pnl_col is not None:
            g["pnl"] = _to_numeric(g[pnl_col])
            g = g.dropna(subset=["pnl"])
            if g.empty:
                continue
            # Strategy files often contain dollars, not capital. Build an equity curve from the user-entered capital.
            g["balance"] = starting_balance_for_pnl + g["pnl"].cumsum()
            g["return"] = g["pnl"] / g["balance"].shift(1)
            notes.append(f"Actual returns calculated from `{pnl_col}` divided by reconstructed prior strategy equity.")

        else:
            continue

        daily = g[["date", "balance", "return"]].copy()
        daily["deposit_withdrawal"] = 0.0
        daily["return"] = daily["return"].replace([np.inf, -np.inf], np.nan)
        daily = daily.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        scopes.append(ScopeData(name=str(strategy_name), source=file_name, daily=daily, notes=notes))

    # If a strategy column exists and there is P/L, also create a combined strategy P/L scope.
    if strategy_col is not None and pnl_col is not None:
        combo = work.copy()
        combo["pnl"] = _to_numeric(combo[pnl_col])
        combo = combo.dropna(subset=["pnl"])
        if not combo.empty:
            combined = combo.groupby(combo["date"].dt.date, as_index=False)["pnl"].sum()
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date")
            combined["balance"] = starting_balance_for_pnl + combined["pnl"].cumsum()
            combined["return"] = combined["pnl"] / combined["balance"].shift(1)
            combined["deposit_withdrawal"] = 0.0
            scopes.insert(
                0,
                ScopeData(
                    name="All Strategies — Combined Actual Returns",
                    source=file_name,
                    daily=combined[["date", "balance", "return", "deposit_withdrawal"]],
                    notes=[f"Combined strategy P/L from `{file_name}` using `{pnl_col}`."],
                ),
            )

    return scopes


def parse_files(files, starting_balance_for_pnl: float) -> Tuple[List[ScopeData], List[str]]:
    scopes: List[ScopeData] = []
    notes: List[str] = []

    for file in files:
        try:
            df = _clean_columns(_read_uploaded_csv(file))
        except Exception as exc:
            notes.append(f"Skipped `{file.name}` because it could not be read: {exc}")
            continue

        portfolio_scope = _daily_from_portfolio_balance_history(df, file.name)
        if portfolio_scope is not None:
            scopes.append(portfolio_scope)
            notes.extend(portfolio_scope.notes)
            # Still parse strategy scopes if the same file has a strategy column.
            strategy_col = _find_column(df, STRATEGY_CANDIDATES)
            if strategy_col is not None:
                extra_scopes = _daily_from_generic_file(df, file.name, starting_balance_for_pnl)
                scopes.extend(extra_scopes)
                for s in extra_scopes:
                    notes.extend(s.notes)
            continue

        generic_scopes = _daily_from_generic_file(df, file.name, starting_balance_for_pnl)
        if generic_scopes:
            scopes.extend(generic_scopes)
            for s in generic_scopes:
                notes.extend(s.notes)
        else:
            notes.append(f"Skipped `{file.name}` because no usable date + balance/return/P&L structure was detected.")

    # De-duplicate names for display.
    seen: Dict[str, int] = {}
    for scope in scopes:
        base = scope.name
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            scope.name = f"{base} ({seen[base]})"

    return scopes, notes


# -----------------------------
# Monte Carlo helpers
# -----------------------------
def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2%}"


def prepare_returns(
    daily: pd.DataFrame,
    weekdays_only: bool,
    exclude_zero_returns: bool,
    exclude_deposit_days: bool,
) -> pd.Series:
    work = daily.copy().sort_values("date")

    if weekdays_only:
        work = work[work["date"].dt.weekday < 5]

    if exclude_deposit_days and "deposit_withdrawal" in work.columns:
        work = work[work["deposit_withdrawal"].fillna(0.0).abs() == 0]

    returns = work["return"].replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    if exclude_zero_returns:
        returns = returns[returns != 0]

    return returns


def monte_carlo_paths(
    returns: pd.Series,
    starting_balance: float,
    years: int,
    simulations: int,
    seed: int,
) -> np.ndarray:
    clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        raise ValueError("No valid return observations are available for Monte Carlo simulation.")

    trading_days = int(252 * years)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean.values, size=(simulations, trading_days), replace=True)
    return starting_balance * np.cumprod(1 + sampled, axis=1)


def path_stats(paths: np.ndarray) -> Dict[str, float]:
    ending = paths[:, -1]
    return {
        "p5": float(np.percentile(ending, 5)),
        "p25": float(np.percentile(ending, 25)),
        "median": float(np.percentile(ending, 50)),
        "p75": float(np.percentile(ending, 75)),
        "p95": float(np.percentile(ending, 95)),
        "mean": float(np.mean(ending)),
    }


def make_historical_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["balance"],
            mode="lines",
            name="Historical equity",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Historical Equity Curve",
        xaxis_title="Date",
        yaxis_title="Account value",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_classic_mc_chart(paths: np.ndarray, current_balance: float, years: int, sample_paths: int, seed: int) -> go.Figure:
    x = np.arange(1, paths.shape[1] + 1) / 252.0
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    rng = np.random.default_rng(seed + years)
    sample_n = min(sample_paths, paths.shape[0])
    if sample_n > 0:
        idx = rng.choice(paths.shape[0], size=sample_n, replace=False)
        for i in idx:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=paths[i],
                    mode="lines",
                    line=dict(width=0.5),
                    opacity=0.08,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # 5th to 95th cone
    fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p5,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            opacity=0.18,
            name="5th–95th percentile range",
            hoverinfo="skip",
        )
    )

    # 25th to 75th cone
    fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p25,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            opacity=0.30,
            name="25th–75th percentile range",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=p50,
            mode="lines",
            line=dict(width=3),
            name="Median projection",
            hovertemplate="Year %{x:.2f}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=current_balance,
        line_dash="dash",
        annotation_text="Current balance",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Monte Carlo Projection — {years}-Year Strategy Distribution",
        xaxis_title="Years",
        yaxis_title="Projected account value",
        hovermode="x unified",
        legend_title_text="Series",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def smooth_density(values: np.ndarray, bins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return values, np.zeros_like(values)

    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    # Simple Gaussian-like smoothing without scipy.
    kernel_x = np.linspace(-3, 3, 17)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel = kernel / kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return centers, smooth


def make_distribution_chart(paths: np.ndarray, years: int) -> go.Figure:
    ending = paths[:, -1]
    x, y = smooth_density(ending)
    stats = path_stats(paths)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="tozeroy",
            name="Ending value distribution",
            hovertemplate="$%{x:,.0f}<br>Density %{y:.6f}<extra></extra>",
        )
    )

    for label, value in [("p5", stats["p5"]), ("median", stats["median"]), ("p95", stats["p95"] )]:
        fig.add_vline(x=value, line_dash="dash", annotation_text=label.upper(), annotation_position="top")

    fig.update_layout(
        title=f"{years}-Year Ending Value Distribution",
        xaxis_title="Ending account value",
        yaxis_title="Probability density",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    return fig


def make_return_distribution_chart(returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, histnorm="probability density", name="Actual daily returns"))
    fig.update_layout(
        title="Actual Daily Return Distribution",
        xaxis_title="Daily return",
        yaxis_title="Probability density",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickformat=".2%")
    return fig


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    uploaded_files = st.file_uploader(
        "Upload balance history / return / strategy P&L CSV",
        type=["csv"],
        accept_multiple_files=True,
        help="Supports portfolio balance-history files with Date/NLV/Day_PL and strategy files with Date + Strategy + P&L, return, or balance columns.",
    )

    starting_balance_for_pnl = st.number_input(
        "Capital basis for P&L-only strategy files",
        min_value=1.0,
        value=100000.0,
        step=5000.0,
        format="%.2f",
    )

    st.divider()
    st.header("Return filters")
    weekdays_only = st.checkbox("Use weekdays only", value=True)
    exclude_zero_returns = st.checkbox("Exclude zero-return days", value=True)
    exclude_deposit_days = st.checkbox("Exclude deposit/withdrawal days", value=True)

    st.divider()
    st.header("Monte Carlo")
    simulations = st.slider("Simulations", min_value=250, max_value=5000, value=1000, step=250)
    sample_paths = st.slider("Faint sample paths shown", min_value=0, max_value=250, value=80, step=10)
    random_seed = st.number_input("Random seed", value=42, step=1)


if not uploaded_files:
    st.info("Upload your balance-history CSV or strategy CSV to generate the Monte Carlo analysis.")
    st.stop()

scopes, import_notes = parse_files(uploaded_files, starting_balance_for_pnl)

if import_notes:
    with st.expander("Import notes", expanded=False):
        for note in import_notes:
            st.write(f"- {note}")

if not scopes:
    st.error("No usable data could be parsed from the uploaded file(s).")
    st.stop()

scope_names = [s.name for s in scopes]
selected_name = st.sidebar.selectbox("Analysis scope", options=scope_names, index=0)
selected_scope = next(s for s in scopes if s.name == selected_name)

daily = selected_scope.daily.copy().sort_values("date").reset_index(drop=True)
clean_returns = prepare_returns(
    daily,
    weekdays_only=weekdays_only,
    exclude_zero_returns=exclude_zero_returns,
    exclude_deposit_days=exclude_deposit_days,
)

if len(clean_returns) < 5:
    st.error("There are fewer than 5 usable return observations after filters. Loosen the filters or upload more data.")
    st.stop()

current_balance = float(daily["balance"].dropna().iloc[-1])
starting_balance = float(daily["balance"].dropna().iloc[0])

# -----------------------------
# Historical section
# -----------------------------
st.subheader(f"Historical Performance — {selected_name}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Starting value", money(starting_balance))
col2.metric("Current value", money(current_balance))
col3.metric("Historical return", pct(current_balance / starting_balance - 1))
col4.metric("Return observations", f"{len(clean_returns):,}")

st.plotly_chart(make_historical_chart(daily), use_container_width=True)

# -----------------------------
# Return diagnostics
# -----------------------------
st.subheader("Actual Return Diagnostics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average daily return", pct(clean_returns.mean()))
col2.metric("Daily volatility", pct(clean_returns.std()))
col3.metric("Best day", pct(clean_returns.max()))
col4.metric("Worst day", pct(clean_returns.min()))

st.plotly_chart(make_return_distribution_chart(clean_returns), use_container_width=True)

with st.expander("Return stream used for Monte Carlo", expanded=False):
    preview = pd.DataFrame({"return": clean_returns}).reset_index(drop=True)
    st.dataframe(preview.tail(25).style.format({"return": "{:.4%}"}), use_container_width=True, hide_index=True)

# -----------------------------
# Monte Carlo section
# -----------------------------
st.subheader("Monte Carlo Projections Based on Actual Returns")
st.markdown(
    "The projection resamples the actual historical daily return stream with replacement. No sit-out adjustment is applied."
)

try:
    paths_1yr = monte_carlo_paths(clean_returns, current_balance, years=1, simulations=simulations, seed=int(random_seed))
    paths_10yr = monte_carlo_paths(clean_returns, current_balance, years=10, simulations=simulations, seed=int(random_seed))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

stats_1 = path_stats(paths_1yr)
stats_10 = path_stats(paths_10yr)

# 1 year
st.markdown("### 1-Year Projection")
st.plotly_chart(make_classic_mc_chart(paths_1yr, current_balance, years=1, sample_paths=sample_paths, seed=int(random_seed)), use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("5th percentile", money(stats_1["p5"]))
c2.metric("Median", money(stats_1["median"]))
c3.metric("95th percentile", money(stats_1["p95"]))
c4.metric("Mean", money(stats_1["mean"]))

st.plotly_chart(make_distribution_chart(paths_1yr, years=1), use_container_width=True)

# 10 year
st.markdown("### 10-Year Projection")
st.plotly_chart(make_classic_mc_chart(paths_10yr, current_balance, years=10, sample_paths=sample_paths, seed=int(random_seed)), use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("5th percentile", money(stats_10["p5"]))
c2.metric("Median", money(stats_10["median"]))
c3.metric("95th percentile", money(stats_10["p95"]))
c4.metric("Mean", money(stats_10["mean"]))

st.plotly_chart(make_distribution_chart(paths_10yr, years=10), use_container_width=True)

summary = pd.DataFrame(
    [
        {"Horizon": "1 Year", "5th Percentile": stats_1["p5"], "25th Percentile": stats_1["p25"], "Median": stats_1["median"], "75th Percentile": stats_1["p75"], "95th Percentile": stats_1["p95"], "Mean": stats_1["mean"]},
        {"Horizon": "10 Years", "5th Percentile": stats_10["p5"], "25th Percentile": stats_10["p25"], "Median": stats_10["median"], "75th Percentile": stats_10["p75"], "95th Percentile": stats_10["p95"], "Mean": stats_10["mean"]},
    ]
)

st.subheader("Projection Summary")
currency_cols = ["5th Percentile", "25th Percentile", "Median", "75th Percentile", "95th Percentile", "Mean"]
st.dataframe(summary.style.format({col: "${:,.0f}" for col in currency_cols}), use_container_width=True, hide_index=True)

st.caption(
    "Monte Carlo results are generated by random resampling of realized daily returns. They are not predictions, guarantees, or investment advice."
)
