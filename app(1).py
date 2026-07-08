import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# ALGO Edge Performance History
# Sit-out overlay + 1-year and 10-year Monte Carlo projections
# ============================================================

st.set_page_config(
    page_title="ALGO Edge Performance History",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Styling / page copy
# -----------------------------
st.title("ALGO Edge Performance History")
st.caption(
    "Historical equity curve, three-month sit-out counterfactual, and probabilistic 1-year / 10-year projections."
)

st.markdown(
    """
The sit-out overlay treats sitting out as a **cash position**, not as a haircut to returns.
During the selected sit-out window, strategy returns are set to **0%**. After that window,
normal compounding resumes.
"""
)


# -----------------------------
# Data helpers
# -----------------------------
DATE_CANDIDATES = [
    "date", "Date", "DATE", "datetime", "Datetime", "timestamp", "Timestamp",
    "time", "Time", "close_time", "Close Time", "Trade Date", "trade_date",
]

BALANCE_CANDIDATES = [
    "balance", "Balance", "BALANCE", "equity", "Equity", "account_value", "Account Value",
    "net_liq", "Net Liq", "NetLiquidation", "Net Liquidation", "value", "Value",
    "cumulative_balance", "Cumulative Balance", "ending_balance", "Ending Balance",
]

RETURN_CANDIDATES = [
    "return", "Return", "returns", "Returns", "daily_return", "Daily Return",
    "pct_return", "Pct Return", "% Return", "percent_return", "Percent Return",
]

PNL_CANDIDATES = [
    "pnl", "PnL", "P/L", "p/l", "profit", "Profit", "daily_pnl", "Daily PnL",
    "net_profit", "Net Profit", "Net P/L", "Realized P/L", "realized_pnl",
]

STRATEGY_CANDIDATES = [
    "strategy", "Strategy", "STRATEGY", "system", "System", "name", "Name",
]


@dataclass
class ParsedData:
    raw: pd.DataFrame
    daily: pd.DataFrame
    source_notes: List[str]


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

    # Fuzzy fallback: useful for broker exports with labels like "Account Balance ($)"
    for col in df.columns:
        normalized = col.lower().replace("_", " ").replace("-", " ")
        for candidate in candidates:
            c = candidate.lower().replace("_", " ").replace("-", " ")
            if c in normalized:
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
    # Try standard CSV first; fall back to Python engine for odd broker exports.
    content = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.BytesIO(content), engine="python")


def parse_uploaded_files(files, starting_balance_for_pnl: float) -> ParsedData:
    frames: List[pd.DataFrame] = []
    notes: List[str] = []

    for file in files:
        df = _clean_columns(_read_uploaded_csv(file))
        df["__source_file"] = file.name

        date_col = _find_column(df, DATE_CANDIDATES)
        if date_col is None:
            notes.append(f"Skipped `{file.name}` because no date column was detected.")
            continue

        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["date"]).copy()
        if df.empty:
            notes.append(f"Skipped `{file.name}` because dates could not be parsed.")
            continue

        balance_col = _find_column(df, BALANCE_CANDIDATES)
        return_col = _find_column(df, RETURN_CANDIDATES)
        pnl_col = _find_column(df, PNL_CANDIDATES)
        strategy_col = _find_column(df, STRATEGY_CANDIDATES)

        if strategy_col is not None:
            df["strategy"] = df[strategy_col].astype(str)
        else:
            df["strategy"] = file.name.rsplit(".", 1)[0]

        if balance_col is not None:
            df["balance"] = _to_numeric(df[balance_col])
            df = df.dropna(subset=["balance"])
            if df.empty:
                notes.append(f"Skipped `{file.name}` because the balance column had no numeric values.")
                continue
            df["input_type"] = "balance"
            notes.append(f"Loaded `{file.name}` using `{date_col}` and balance column `{balance_col}`.")

        elif return_col is not None:
            df["return"] = _to_numeric(df[return_col])
            # If returns are provided as percentages, convert to decimals.
            if df["return"].abs().median(skipna=True) > 1:
                df["return"] = df["return"] / 100.0
            df = df.dropna(subset=["return"])
            if df.empty:
                notes.append(f"Skipped `{file.name}` because the return column had no numeric values.")
                continue
            df["balance"] = starting_balance_for_pnl * (1 + df.sort_values("date")["return"]).cumprod()
            df["input_type"] = "return"
            notes.append(f"Loaded `{file.name}` using `{date_col}` and return column `{return_col}`.")

        elif pnl_col is not None:
            df["pnl"] = _to_numeric(df[pnl_col])
            df = df.dropna(subset=["pnl"])
            if df.empty:
                notes.append(f"Skipped `{file.name}` because the P/L column had no numeric values.")
                continue
            df = df.sort_values("date")
            df["balance"] = starting_balance_for_pnl + df["pnl"].cumsum()
            df["input_type"] = "pnl"
            notes.append(
                f"Loaded `{file.name}` using `{date_col}` and P/L column `{pnl_col}`. "
                f"Balance was reconstructed from the starting balance input."
            )
        else:
            notes.append(
                f"Skipped `{file.name}` because no balance, return, or P/L column was detected."
            )
            continue

        frames.append(df)

    if not frames:
        return ParsedData(raw=pd.DataFrame(), daily=pd.DataFrame(), source_notes=notes)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values("date")

    # If files provide actual account balances, the most recent balance per day is the cleanest daily series.
    # If multiple strategy files are uploaded, this still avoids adding account balances together.
    daily = (
        raw.sort_values("date")
        .groupby(raw["date"].dt.date, as_index=False)
        .tail(1)[["date", "balance"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.drop_duplicates(subset=["date"], keep="last")
    daily["return"] = daily["balance"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return ParsedData(raw=raw, daily=daily, source_notes=notes)


# -----------------------------
# Modeling helpers
# -----------------------------
def build_equity_curve(returns: pd.Series, starting_balance: float) -> pd.Series:
    returns = returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return starting_balance * (1 + returns).cumprod()


def apply_sitout_window(
    returns: pd.Series,
    dates: pd.Series,
    start_date: pd.Timestamp,
    months: int = 3,
) -> Tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    adjusted = returns.copy().fillna(0.0)
    sitout_start = pd.Timestamp(start_date).normalize()
    sitout_end = sitout_start + pd.DateOffset(months=months)
    mask = (pd.to_datetime(dates) >= sitout_start) & (pd.to_datetime(dates) < sitout_end)
    adjusted.loc[mask.values] = 0.0
    return adjusted, sitout_start, sitout_end


def describe_gap(full_final: float, sitout_final: float) -> Tuple[str, float, float]:
    gap = full_final - sitout_final
    pct = gap / full_final if full_final else np.nan
    if gap > 0:
        label = "Opportunity cost of sitting out"
    elif gap < 0:
        label = "Capital protected by sitting out"
    else:
        label = "No difference"
    return label, gap, pct


def make_historical_overlay_chart(
    dates: pd.Series,
    full_curve: pd.Series,
    sitout_curve: pd.Series,
    sitout_start: pd.Timestamp,
    sitout_end: pd.Timestamp,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=full_curve,
            mode="lines",
            name="Full Participation",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sitout_curve,
            mode="lines",
            name="Sit Out Window",
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
        title="Historical Overlay: Full Participation vs. Sitting Out",
        xaxis_title="Date",
        yaxis_title="Account Value",
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
        historical_returns.dropna()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    # Ignore the artificial first 0% return when possible.
    if len(clean_returns) > 1 and clean_returns.iloc[0] == 0:
        clean_returns = clean_returns.iloc[1:]

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


def make_projection_chart(
    full_paths: np.ndarray,
    sitout_paths: np.ndarray,
    start_date: pd.Timestamp,
    years: int,
) -> go.Figure:
    dates = pd.bdate_range(start=start_date + pd.offsets.BDay(1), periods=full_paths.shape[1])
    full = summarize_paths(full_paths)
    sitout = summarize_paths(sitout_paths)
    full["date"] = dates
    sitout["date"] = dates

    fig = go.Figure()

    # Full participation percentile band
    fig.add_trace(
        go.Scatter(
            x=full["date"], y=full["p95"], mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip", name="Full p95"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=full["date"], y=full["p5"], mode="lines", line=dict(width=0),
            fill="tonexty", opacity=0.15, name="Full 5th-95th Percentile",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=full["date"], y=full["p50"], mode="lines",
            name="Full Participation Median",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    # Sit-out median
    fig.add_trace(
        go.Scatter(
            x=sitout["date"], y=sitout["p50"], mode="lines",
            name=f"Sit Out First 3 Months Median",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{years}-Year Projection Overlay",
        xaxis_title="Date",
        yaxis_title="Projected Account Value",
        hovermode="x unified",
        legend_title_text="Scenario",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_distribution_chart(
    full_paths: np.ndarray,
    sitout_paths: np.ndarray,
    years: int,
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
            name="Sit Out First 3 Months",
            opacity=0.6,
            nbinsx=60,
        )
    )
    fig.update_layout(
        title=f"{years}-Year Ending Value Distribution",
        xaxis_title="Ending Account Value",
        yaxis_title="Probability Density",
        barmode="overlay",
        legend_title_text="Scenario",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


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


def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2%}"


# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    uploaded_files = st.file_uploader(
        "Upload balance history / returns / P&L CSV",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Preferred: a CSV with Date and Balance columns. Also supports Date + Return, "
            "or Date + P/L using the starting balance below."
        ),
    )

    starting_balance_for_pnl = st.number_input(
        "Starting balance for P/L-only files",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
        format="%.2f",
    )

    st.divider()
    st.subheader("Sit-out overlay")
    sitout_months = st.slider("Sit-out months", min_value=1, max_value=12, value=3, step=1)

    st.divider()
    st.subheader("Monte Carlo")
    simulations = st.slider("Simulations", min_value=250, max_value=5000, value=1000, step=250)
    random_seed = st.number_input("Random seed", value=42, step=1)


if not uploaded_files:
    st.info(
        "Upload your historical balance CSV to generate the overlays. "
        "The cleanest file format is `date, balance`."
    )
    st.stop()

parsed = parse_uploaded_files(uploaded_files, starting_balance_for_pnl=starting_balance_for_pnl)

if parsed.source_notes:
    with st.expander("Import notes", expanded=False):
        for note in parsed.source_notes:
            st.write(f"- {note}")

if parsed.daily.empty or len(parsed.daily) < 3:
    st.error("I could not build a daily equity curve from the uploaded files. Check for a date column and a balance, return, or P/L column.")
    st.stop()

# -----------------------------
# Historical data preparation
# -----------------------------
daily = parsed.daily.copy().sort_values("date").reset_index(drop=True)
daily["return"] = daily["balance"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

first_date = daily["date"].min().normalize()
last_date = daily["date"].max().normalize()
default_sitout_start = max(first_date, last_date - pd.DateOffset(months=sitout_months))

with st.sidebar:
    sitout_start_date = st.date_input(
        "Sit-out start date",
        value=default_sitout_start.date(),
        min_value=first_date.date(),
        max_value=last_date.date(),
    )

sitout_returns, sitout_start, sitout_end = apply_sitout_window(
    returns=daily["return"],
    dates=daily["date"],
    start_date=pd.Timestamp(sitout_start_date),
    months=sitout_months,
)

starting_balance = float(daily["balance"].iloc[0])
full_curve = build_equity_curve(daily["return"], starting_balance)
sitout_curve = build_equity_curve(sitout_returns, starting_balance)

full_final = float(full_curve.iloc[-1])
sitout_final = float(sitout_curve.iloc[-1])
gap_label, gap_value, gap_pct = describe_gap(full_final, sitout_final)

# -----------------------------
# Current performance summary
# -----------------------------
st.subheader("Historical Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Start", money(float(full_curve.iloc[0])))
col2.metric("Current", money(full_final))
col3.metric("Total Return", pct(full_final / float(full_curve.iloc[0]) - 1))
col4.metric("Daily Observations", f"{len(daily):,}")

fig_hist = make_historical_overlay_chart(
    dates=daily["date"],
    full_curve=full_curve,
    sitout_curve=sitout_curve,
    sitout_start=sitout_start,
    sitout_end=sitout_end,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Three-Month Sit-Out Analysis")
col1, col2, col3 = st.columns(3)
col1.metric("Full Participation Final Value", money(full_final))
col2.metric("Sit-Out Final Value", money(sitout_final))
col3.metric(gap_label, money(abs(gap_value)), pct(abs(gap_pct)))

if gap_value > 0:
    st.success(
        "In this selected window, sitting out reduced ending capital versus full participation. "
        "That is the opportunity cost of missing the positive-drift return stream."
    )
elif gap_value < 0:
    st.warning(
        "In this selected window, sitting out improved ending capital versus full participation. "
        "That means the avoided losses were larger than the missed gains."
    )
else:
    st.info("In this selected window, the sit-out and full-participation paths ended at the same value.")

with st.expander("What this overlay is actually testing", expanded=False):
    st.markdown(
        f"""
- **Full Participation:** every historical daily return is included.
- **Sit Out:** returns from **{sitout_start.date()} through {(sitout_end - pd.Timedelta(days=1)).date()}** are set to **0%**.
- **Re-entry:** after the sit-out window, the model resumes the same daily return sequence.
- **Interpretation:** if the sit-out line finishes higher, sitting out protected capital. If it finishes lower, sitting out created opportunity cost.
"""
    )

# -----------------------------
# Return diagnostics
# -----------------------------
st.subheader("Return Diagnostics")
clean_returns = daily["return"].iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Daily Return", pct(clean_returns.mean()))
col2.metric("Daily Volatility", pct(clean_returns.std()))
col3.metric("Best Day", pct(clean_returns.max()))
col4.metric("Worst Day", pct(clean_returns.min()))

fig_ret = go.Figure()
fig_ret.add_trace(
    go.Histogram(x=clean_returns, nbinsx=50, name="Daily Returns", histnorm="probability density")
)
fig_ret.update_layout(
    title="Historical Daily Return Distribution",
    xaxis_title="Daily Return",
    yaxis_title="Probability Density",
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig_ret, use_container_width=True)

# -----------------------------
# Projections
# -----------------------------
st.subheader("Projection Overlays")
st.markdown(
    "The projection overlay compares normal compounding against a scenario where the account sits in cash for the first three months, then resumes trading."
)

current_balance = float(daily["balance"].iloc[-1])

try:
    paths_1yr_full = monte_carlo_paths(
        clean_returns,
        starting_balance=current_balance,
        years=1,
        simulations=simulations,
        sitout_months=0,
        seed=int(random_seed),
    )
    paths_1yr_sitout = monte_carlo_paths(
        clean_returns,
        starting_balance=current_balance,
        years=1,
        simulations=simulations,
        sitout_months=sitout_months,
        seed=int(random_seed),
    )
    paths_10yr_full = monte_carlo_paths(
        clean_returns,
        starting_balance=current_balance,
        years=10,
        simulations=simulations,
        sitout_months=0,
        seed=int(random_seed),
    )
    paths_10yr_sitout = monte_carlo_paths(
        clean_returns,
        starting_balance=current_balance,
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
fig_1yr = make_projection_chart(paths_1yr_full, paths_1yr_sitout, last_date, years=1)
st.plotly_chart(fig_1yr, use_container_width=True)

stats_1_full = final_stats(paths_1yr_full)
stats_1_sit = final_stats(paths_1yr_sitout)
median_gap_1 = stats_1_full["median"] - stats_1_sit["median"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Full Median", money(stats_1_full["median"]))
c2.metric("Sit-Out Median", money(stats_1_sit["median"]))
c3.metric("Median Opportunity Cost / Benefit", money(abs(median_gap_1)))
c4.metric("Full 5th Percentile", money(stats_1_full["p5"]))

fig_1yr_dist = make_distribution_chart(paths_1yr_full, paths_1yr_sitout, years=1)
st.plotly_chart(fig_1yr_dist, use_container_width=True)

# 10-year projection
st.markdown("### 10-Year Projection")
fig_10yr = make_projection_chart(paths_10yr_full, paths_10yr_sitout, last_date, years=10)
st.plotly_chart(fig_10yr, use_container_width=True)

stats_10_full = final_stats(paths_10yr_full)
stats_10_sit = final_stats(paths_10yr_sitout)
median_gap_10 = stats_10_full["median"] - stats_10_sit["median"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Full Median", money(stats_10_full["median"]))
c2.metric("Sit-Out Median", money(stats_10_sit["median"]))
c3.metric("Median Opportunity Cost / Benefit", money(abs(median_gap_10)))
c4.metric("Full 5th Percentile", money(stats_10_full["p5"]))

fig_10yr_dist = make_distribution_chart(paths_10yr_full, paths_10yr_sitout, years=10)
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

currency_cols = ["5th Percentile", "25th Percentile", "Median", "75th Percentile", "95th Percentile"]
st.dataframe(
    summary_df.style.format({col: "${:,.0f}" for col in currency_cols}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Monte Carlo projections are based on random resampling of the historical daily return stream. "
    "They are not predictions; they are a way to visualize the distribution of possible outcomes if the historical return/volatility profile persists."
)
