"""
Streamlit presentation of the stored out-of-sample evaluation results.

All values come from the saved artifacts loaded by
``src.forecasting.evaluation.artifacts``; nothing here runs an experiment or
touches the network.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecasting.evaluation.artifacts import (
    ALL_HORIZONS,
    DEFAULT_RESULTS_DIR,
    EvaluationArtifacts,
    best_models,
    best_models_by_horizon,
    bias_comparison,
    download_files,
    fixed_nudge_comparison,
    indicator_comparison,
    load_evaluation_artifacts,
    metric_by_horizon,
    metric_table,
    period_errors,
)

MODEL_COLORS = {
    "Naive (last value)": "#8D99AE",
    "3-yr CAGR": "#E76F51",
    "Damped ETS": "#2E86AB",
    "Logistic Growth": "#2A9D8F",
}
METRIC_TITLES = {"mape": "MAPE (%)", "rmse": "RMSE (US$ millions)", "mae": "MAE (US$ millions)"}
MONEY = "{:,.0f}"
PCT = "{:.2f}%"


def _artifact_mtimes(results_dir: Path) -> tuple:
    if not results_dir.exists():
        return ()
    return tuple(sorted((str(p), p.stat().st_mtime) for p in results_dir.rglob("*") if p.is_file()))


@st.cache_data(show_spinner=False)
def _cached_load(_mtimes: tuple) -> EvaluationArtifacts:
    return load_evaluation_artifacts()


def load_artifacts() -> EvaluationArtifacts:
    """Load stored artifacts, re-reading only when a result file changes."""
    return _cached_load(_artifact_mtimes(DEFAULT_RESULTS_DIR))


def _layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Forecast horizon",
        yaxis_title=y_title,
        height=380,
        margin=dict(l=10, r=10, t=45, b=95),
        legend=dict(orientation="h", yanchor="top", y=-0.28, x=0),
        template="plotly_white",
    )
    return fig


def metric_horizon_chart(pivot: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    for model in pivot.columns:
        fig.add_trace(go.Scatter(
            x=list(pivot.index), y=pivot[model], mode="lines+markers", name=model,
            line=dict(color=MODEL_COLORS.get(model), width=2.5), marker=dict(size=7),
        ))
    label = metric.upper()
    return _layout(fig, f"{label} by horizon (pooled)", METRIC_TITLES[metric])


def baseline_vs_adjusted_chart(table: pd.DataFrame, metric_label: str, horizon_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["Model"], y=table[f"Baseline {metric_label}"], name="Baseline",
                         marker_color="#8D99AE"))
    fig.add_trace(go.Bar(x=table["Model"], y=table[f"Adjusted {metric_label}"],
                         name="Indicator-adjusted (past-only)", marker_color="#2E86AB"))
    fig.update_layout(barmode="group")
    _layout(fig, f"Baseline vs indicator-adjusted {metric_label} ({horizon_label})", f"{metric_label} (%)")
    fig.update_xaxes(title_text="")
    return fig


def period_chart(table: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    for model, rows in table.groupby("Model", sort=False):
        fig.add_trace(go.Bar(x=rows["Target years"], y=rows["MAPE"], name=model,
                             marker_color=MODEL_COLORS.get(model)))
    fig.update_layout(barmode="group")
    _layout(fig, f"MAPE by target period ({horizon}Y ahead)", "MAPE (%)")
    fig.update_xaxes(title_text="Target years")
    return fig


def adjustment_factor_chart(by_origin: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=by_origin["origin"], y=by_origin["applied_adjustment_pct"], mode="lines+markers",
        line=dict(color="#2E86AB", width=2), name="Applied adjustment",
    ))
    _layout(fig, "Applied indicator adjustment by forecast origin", "Uplift applied to forecast (%)")
    fig.update_xaxes(title_text="Forecast origin (last training year)")
    fig.update_layout(showlegend=False, height=280)
    return fig


def _highlight_min_by_horizon(table: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    for _, group in table.groupby("Horizon"):
        for column in columns:
            values = group[column].abs() if column == "Bias" else group[column]
            best = values[values == values.min()].index
            styles.loc[best, column] = "background-color: rgba(42, 157, 143, 0.22); font-weight: 600"
    return styles


def render_summary(artifacts: EvaluationArtifacts) -> None:
    meta = artifacts.baseline["metadata"]
    origins = meta["origins"]
    horizons = ", ".join(str(h) for h in meta["evaluated_horizons"])
    st.markdown(f"**Dataset:** {meta['dataset_name']} &nbsp;·&nbsp; **Source:** {meta['source']} "
                f"&nbsp;·&nbsp; **Label:** {meta['data_label']}")
    cols = st.columns(5)
    cols[0].metric("Period", f"{meta['first_year']}–{meta['last_year']}")
    cols[1].metric("Observations", meta["n_observations"])
    cols[2].metric("Walk-forward origins", f"{origins[0]}–{origins[-1]}", help=f"{meta['n_folds']} folds")
    cols[3].metric("Horizons (years)", horizons)
    cols[4].metric("Models", len(meta["models"]), help=", ".join(meta["models"]))
    st.caption("Evaluation: expanding-window walk-forward, out-of-sample. Models: "
               + ", ".join(meta["models"]) + ".")
    st.warning("**Current-vintage data — not a real-time vintage backtest.** " + meta["current_vintage_status"])


def render_baseline_section(artifacts: EvaluationArtifacts) -> None:
    summaries = artifacts.baseline["summaries"]
    report = artifacts.baseline["report_pooled"]
    st.header("Baseline model comparison")

    best = best_models(report)
    cols = st.columns(3)
    for col, metric in zip(cols, ("mape", "rmse", "mae")):
        entry = best.get(metric)
        if entry:
            value = PCT.format(entry["value"]) if metric == "mape" else MONEY.format(entry["value"])
            col.metric(f"Lowest pooled {metric.upper()} (all horizons)", entry["model"])
            col.caption(f"{metric.upper()} = {value}, coverage {entry['coverage']:.0f}%")
    by_horizon = best_models_by_horizon(report)
    if by_horizon:
        st.caption("Lowest pooled MAPE by horizon: " + " · ".join(
            f"{h}: {e['model']} ({e['value']:.2f}%)" for h, e in by_horizon.items())
            + ". Different metrics favour different models, so no overall winner is declared.")

    aggregation = st.radio("Aggregation", ["pooled", "fold_mean"], horizontal=True,
                           format_func=lambda a: "Pooled" if a == "pooled" else "Fold-mean",
                           key="eval_aggregation")
    table = metric_table(summaries, aggregation)
    styler = (table.style
              .apply(lambda _: _highlight_min_by_horizon(table, ["MAPE", "RMSE", "MAE", "Bias"]), axis=None)
              .format({"MAPE": PCT, "RMSE": MONEY, "MAE": MONEY, "Bias": "{:+,.0f}",
                       "Abs bias": MONEY, "Coverage": "{:.0f}%"}))
    st.dataframe(styler, hide_index=True, width="stretch")
    st.caption("RMSE, MAE and bias in US$ millions (bias = mean forecast − actual; negative = under-forecast). "
               "Shaded cells: lowest value within each horizon (bias by absolute value). "
               "All models were evaluated on the same origins; coverage is the share of successful folds.")

    cols = st.columns(3)
    for col, metric in zip(cols, ("mape", "rmse", "mae")):
        col.plotly_chart(metric_horizon_chart(metric_by_horizon(summaries, metric), metric),
                         width="stretch", key=f"eval_{metric}_chart")


def _indicator_header(artifacts: EvaluationArtifacts) -> None:
    meta = artifacts.indicator["metadata"]
    provenance = artifacts.indicator_provenance or {}
    weights = meta.get("indicator_weights", {})
    name = provenance.get("name", "BEA Personal Consumption Expenditures — Goods")
    cols = st.columns(3)
    cols[0].markdown(f"**Indicator**  \n{name} (BEA series `{provenance.get('series_code', 'DGDSRC')}`)")
    cols[1].markdown("**Indicator weight**  \n" + ", ".join(f"{w:g}" for w in weights.values()))
    cols[2].markdown(f"**Overall indicator weight**  \n{meta.get('overall_indicator_weight'):g}")
    st.caption("Point-in-time: for each origin Y only indicator observations with year ≤ Y are used. "
               "The production IndicatorAdjustment formula is unchanged.")
    run_log = (artifacts.run_log or {}).get("specifications", {})
    secondary = run_log.get("spec_b_goods_internet")
    if secondary and secondary.get("status") != "run":
        st.caption("Secondary specification (goods + U.S. internet users) was not run: the internet-user "
                   "series failed the data-quality/break inspection (documented source changes).")


def render_indicator_section(artifacts: EvaluationArtifacts) -> None:
    st.header("Indicator Adjustment Evaluation")
    if not artifacts.indicator_available:
        st.warning("Indicator experiment artifacts are unavailable; this section cannot be shown.")
        return
    effects = artifacts.indicator["effects"]
    _indicator_header(artifacts)

    options = [ALL_HORIZONS, "1", "2", "3"]
    horizon = st.radio("Horizon", options, horizontal=True, key="eval_indicator_horizon",
                       format_func=lambda h: "All horizons" if h == ALL_HORIZONS else f"{h}Y")
    horizon_label = "all horizons" if horizon == ALL_HORIZONS else f"{horizon}Y"
    left, right = st.columns([1, 1])
    mape = indicator_comparison(effects, "mape", horizon)
    left.plotly_chart(baseline_vs_adjusted_chart(mape, "MAPE", horizon_label), width="stretch",
                      key="eval_indicator_chart")
    with right:
        tabs = st.tabs(["MAPE", "RMSE", "MAE"])
        for tab, metric in zip(tabs, ("mape", "rmse", "mae")):
            table = indicator_comparison(effects, metric, horizon)
            label = metric.upper()
            fmt = PCT if metric == "mape" else MONEY
            change_fmt = "{:+.2f} pp" if metric == "mape" else "{:+,.0f}"
            tab.dataframe(table.style.format({
                f"Baseline {label}": fmt, f"Adjusted {label}": fmt,
                f"Change {label}": change_fmt, f"Change {label} %": "{:+.1f}%",
            }), hide_index=True, width="stretch")
        st.caption("Change = adjusted − baseline; negative means lower error. Paired on identical forecasts.")
    st.info("The indicator adjustment did not provide consistent predictive improvement. "
            "Its effect was primarily a small upward bias correction.")


def render_bias_section(artifacts: EvaluationArtifacts) -> None:
    if not artifacts.indicator_available:
        return
    st.header("Bias diagnostic")
    effects = artifacts.indicator["effects"]
    bias = bias_comparison(effects)
    st.dataframe(bias.style.format({"Baseline bias": "{:+,.0f}", "Adjusted bias": "{:+,.0f}",
                                    "Bias change": "{:+,.0f}"}), hide_index=True, width="stretch")
    st.markdown(
        "The indicator adjustment behaves primarily as a small upward forecast shift. "
        "Improvements occur mainly where baseline forecasts under-predict: every paired win above "
        "is a forecast the baseline had under-predicted (all horizons, US$ millions).")

    left, right = st.columns([1, 1])
    by_origin = artifacts.indicator["by_origin"]
    left.plotly_chart(adjustment_factor_chart(by_origin), width="stretch", key="eval_factor_chart")
    with right:
        nudge = fixed_nudge_comparison(effects)
        st.dataframe(nudge.style.format({
            "Indicator MAPE": PCT, "Fixed-nudge MAPE": PCT, "Indicator RMSE": MONEY,
            "Fixed-nudge RMSE": MONEY, "Indicator MAE": MONEY, "Fixed-nudge MAE": MONEY,
        }), hide_index=True, width="stretch")
        mean_factor = artifacts.indicator["metadata"].get("mean_adjustment_factor")
        st.caption(
            f"Fixed-nudge diagnostic: every baseline forecast multiplied by the full-sample mean factor "
            f"({mean_factor:.4f}). It slightly outperformed the indicator's origin-varying adjustment, so the "
            "indicator's variation added no measurable information. Diagnostic only — it uses the "
            "full-sample mean and is not point-in-time.")


def render_regime_section(artifacts: EvaluationArtifacts) -> None:
    st.header("Pandemic / regime-change diagnostic")
    breakdown = artifacts.baseline["period_breakdown"]
    horizon = st.radio("Horizon", [1, 2, 3], index=2, horizontal=True, key="eval_period_horizon",
                       format_func=lambda h: f"{h}Y")
    table = period_errors(breakdown, horizon)
    left, right = st.columns([1, 1])
    left.plotly_chart(period_chart(table, horizon), width="stretch", key="eval_period_chart")
    right.dataframe(table.style.format({"MAPE": PCT, "Share of abs. error %": "{:.1f}%"}),
                    hide_index=True, width="stretch")

    one_year = period_errors(breakdown, 1)
    pandemic = one_year[one_year["Target years"] == "2020-2022"]
    cagr = table[(table["Model"] == "3-yr CAGR")].set_index("Target years")["MAPE"]
    notes = []
    if not pandemic.empty:
        share = pandemic.set_index("Model")["Share of abs. error %"]
        trend = share.drop("Naive (last value)", errors="ignore")
        notes.append(f"At 1Y, the three 2020–2022 target years ({int(pandemic['Forecasts'].iloc[0])} of "
                     f"{int(one_year.groupby('Model')['Forecasts'].sum().iloc[0])} forecasts per model) account for "
                     f"{trend.min():.0f}–{trend.max():.0f}% of the trend models' absolute error.")
    if {"2005-2019", "2023-2025"} <= set(cagr.index):
        notes.append(f"3-yr CAGR extrapolated the pandemic surge: at {horizon}Y its MAPE is "
                     f"{cagr['2023-2025']:.1f}% for 2023–2025 targets vs {cagr['2005-2019']:.1f}% for 2005–2019.")
    notes.append("Forecast accuracy deteriorates under regime changes; 2020–2025 are kept unchanged in the "
                 "evaluation (no years removed or altered).")
    st.markdown("\n".join(f"- {n}" for n in notes))


def render_news_section() -> None:
    st.header("Historical News Evaluation")
    st.info("Historical analyzed-news data sufficient for a comparable long-horizon experiment was "
            "unavailable, so the news-adjusted variant was not included in the real-data empirical results.")
    st.caption("News evaluation infrastructure exists, but empirical long-horizon news evaluation was "
               "excluded because suitable historical analyzed-news data was unavailable. This is not "
               "evidence that news adjustments are ineffective.")


def render_methodology() -> None:
    with st.expander("Methodology and reproducibility"):
        st.markdown(
            """
1. **Walk-forward evaluation** — for each origin year Y, models are fitted on data up to Y and forecast
   Y+1…Y+3; the origin then advances one year (expanding window, 19 origins).
2. **Temporal isolation** — training data, indicators and (where used) news are restricted to information
   dated at or before the origin; later values never influence a forecast.
3. **Naive benchmark** — the last observed value repeated; a model is only useful if it beats this.
4. **Forecast horizons** — 1-, 2- and 3-year-ahead errors are reported separately and pooled.
5. **Metrics** — MAPE, RMSE, MAE, bias (mean forecast − actual) and coverage; pooled over all forecasts
   and as fold means. No composite score and no significance testing.
6. **Indicator point-in-time filtering** — the unchanged production indicator adjustment receives only
   indicator observations with year ≤ origin.
7. **Current-vintage limitation** — Census and BEA series are today's revised values, not what a
   forecaster saw at each origin; this is not a real-time vintage backtest.
8. **News-data limitation** — no suitable historical analyzed-news archive exists for 2004–2022, so
   news adjustments are not part of these results.

Full methodology: `docs/evaluation.md`. Raw official sources are in `data/evaluation/raw/`; results in
`data/evaluation/results/` (regenerate with the scripts in `scripts/`, see README).
            """
        )


def render_evaluation_downloads(artifacts: EvaluationArtifacts, key_prefix: str) -> None:
    """Download buttons for the stored evaluation files (written by the evaluation export module)."""
    files = download_files(artifacts)
    if not files:
        st.warning("Empirical evaluation artifacts are unavailable (data/evaluation/results/ not found).")
        return
    cols = st.columns(3)
    for index, item in enumerate(files):
        cols[index % 3].download_button(
            label=item["label"], data=Path(item["path"]).read_bytes(), file_name=item["file_name"],
            mime=item["mime"], key=f"{key_prefix}_{item['key']}", width="stretch",
        )


def render_missing(artifacts: EvaluationArtifacts) -> None:
    st.error("The empirical evaluation artifacts are unavailable, so the stored results cannot be shown. "
             "No results are generated on this page. Regenerate them with "
             "`python scripts/run_census_baseline_experiment.py` and "
             "`python scripts/run_census_indicator_experiment.py`.")
    with st.expander("Missing files"):
        st.code("\n".join(artifacts.missing) or "(none)")
