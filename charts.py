"""Plotly charts for the SHAP explanations, styled for the app's dark theme.

waterfall() is ported from lambda_ml_streamlit_exercise/app/charts.py. guide_waterfall()
follows guide_chart() in geographic_playground_mini/pareto.py: a small made-up example
with numbered badges that match the steps written beside it. icon_array() is ported from
the same app's "Who would we phone?" page.

Red and blue are the dataviz skill's dark-mode diverging poles, validated against the
app background (#1b1a1a). The +/- value labels mean colour is never the only cue.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from calls import GROUPS

UP = "#e66767"  # pushes the risk of not attending up
DOWN = "#3987e5"  # pushes it down
TOTAL = "#c3c2b7"  # the average and this appointment: neutral, not a direction
TEXT = "#f0efec"
MUTED = "#898781"
FONT = dict(family="Space Grotesk, sans-serif", color=TEXT, size=15)


def _waterfall_trace(labels, values, measure, texts):
    return go.Waterfall(
        orientation="h",
        y=labels,
        x=values,
        measure=measure,
        text=texts,
        textposition="outside",
        textfont=dict(color=TEXT),
        increasing=dict(marker=dict(color=UP)),
        decreasing=dict(marker=dict(color=DOWN)),
        totals=dict(marker=dict(color=TOTAL)),
        connector=dict(line=dict(color=MUTED, width=1)),
        hovertemplate="%{y}: %{text}<extra></extra>",
    )


def _layout(fig, height, **kwargs):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        showlegend=False,
        yaxis=dict(autorange="reversed"),
        **kwargs,
    )
    return fig


def waterfall(expl: dict) -> go.Figure:
    """SHAP waterfall in percentage points. Red pushes towards not attending, blue away."""
    steps = list(expl["shown"])
    if expl["other"]:
        steps.append(expl["other"])
    labels = ["Average appointment"] + [s["label"] for s in steps] + ["This appointment"]
    values = [expl["base"]] + [s["value"] for s in steps] + [expl["fx"]]
    measure = ["absolute"] + ["relative"] * len(steps) + ["total"]
    texts = [f"{expl['base']:.1f}%"] + [f"{s['value']:+.1f}" for s in steps] + [f"{expl['fx']:.1f}%"]
    fig = go.Figure(_waterfall_trace(labels, values, measure, texts))
    # Room past the furthest bar end for its outside label
    running = [expl["base"]]
    for s in steps:
        running.append(running[-1] + s["value"])
    x_max = max(running) * 1.15 + 3
    return _layout(
        fig,
        120 + 40 * len(labels),
        margin=dict(l=0, r=50, t=10, b=40),
        xaxis=dict(
            title="Predicted risk of not attending (%). Red raises it, blue lowers it. Changes in percentage points.",
            ticksuffix="%",
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
            range=[min(0, min(running)), x_max],
        ),
    )


FILTERED = "#1fb1b9"  # the theme's primary colour; no direction, so not red or blue


def importance_bars(overall: list, overall_title: str, filtered: list | None = None, filtered_title: str = "") -> go.Figure:
    """Mean |SHAP| per factor (lists from explain.group_importance), with no numbers on show.

    With `filtered`, a second panel sits to the right on the same scale and in the overall
    chart's order, with the factor names shown once, so bar lengths compare row by row. The
    exact figures stay in the hover. No direction, so neutral colours rather than red or blue.
    """
    labels = [i["label"] for i in overall]
    panels = [(overall, overall_title, TOTAL)]
    if filtered is not None:
        by_label = {i["label"]: i["value"] for i in filtered}
        panels.append(([{"label": l, "value": by_label[l]} for l in labels], filtered_title, FILTERED))
    fig = make_subplots(
        rows=1, cols=len(panels), shared_yaxes=True, horizontal_spacing=0.04,
        subplot_titles=[title for _, title, _ in panels],
    )
    for col, (items, _, colour) in enumerate(panels, start=1):
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=labels,
                x=[i["value"] for i in items],
                marker=dict(color=colour),
                hovertemplate="%{y}: moves the risk by %{x:.1f} percentage points on average<extra></extra>",
            ),
            row=1,
            col=col,
        )
    x_max = max(i["value"] for items, _, _ in panels for i in items) * 1.05
    fig.update_xaxes(range=[0, x_max], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    _layout(fig, 100 + 40 * len(labels), margin=dict(l=0, r=10, t=40, b=10))
    # Panel titles left-aligned over their own panel
    for ann, axis in zip(fig.layout.annotations, ["xaxis", "xaxis2"]):
        ann.update(x=fig.layout[axis].domain[0], xanchor="left", font=dict(size=15, color=TEXT))
    return fig


# Made-up example for the "How to read this chart" box: average 30%, three factors, 47%
GUIDE_STEPS = [
    ("Average appointment", 30, "absolute", "30%"),
    ("Missed appointments = 3", 18, "relative", "+18"),
    ("Days since booking = 60", 6, "relative", "+6"),
    ("Age = 70", -7, "relative", "-7"),
    ("This appointment", 47, "total", "47%"),
]
# Which row each numbered badge sits beside (the numbers match the steps written next to it)
GUIDE_BADGES = {1: "Average appointment", 2: "Missed appointments = 3", 3: "This appointment"}


def guide_waterfall() -> go.Figure:
    """Small annotated example waterfall with numbered badges in a column on the right."""
    labels, values, measure, texts = (list(t) for t in zip(*GUIDE_STEPS))
    fig = go.Figure(_waterfall_trace(labels, values, measure, texts))
    badge_x = 64
    fig.add_trace(
        go.Scatter(
            x=[badge_x] * len(GUIDE_BADGES),
            y=list(GUIDE_BADGES.values()),
            mode="markers+text",
            text=[f"<b>{n}</b>" for n in GUIDE_BADGES],
            marker=dict(size=26, color="#ffffff"),
            textfont=dict(color="#1b1a1a", size=15),
            hoverinfo="skip",
        )
    )
    return _layout(
        fig,
        300,
        width=480,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(visible=False, range=[0, badge_x + 4], fixedrange=True),
    )


# The dataviz skill's fixed status steps (good / warning / critical), plus a recessive
# grey for the large "nothing happens" group. Green and red are close for deuteranopes,
# so every group also has its own marker shape, a legend label with its count, a hover
# label and a table view: colour is never the only cue.
_ICON_STYLE = {
    "called_dna": ("#0ca30c", "circle"),
    "called_attended": ("#fab219", "square"),
    "not_called_dna": ("#d03b3b", "x-thin-open"),
    "not_called_attended": (MUTED, "circle-open"),
}
_COLS = 40


def icon_array(outcomes: dict) -> go.Figure:
    """1,000 markers (40 x 25), one per 1,000 past appointments, grouped by outcome."""
    fig = go.Figure()
    start = 0
    for key, label in GROUPS:
        count = outcomes[key]
        idx = range(start, start + count)
        colour, symbol = _ICON_STYLE[key]
        fig.add_trace(
            go.Scatter(
                x=[i % _COLS for i in idx],
                y=[-(i // _COLS) for i in idx],
                mode="markers",
                name=f"{label}: {count}",
                marker=dict(color=colour, symbol=symbol, size=9, line=dict(width=2, color=colour)),
                hovertemplate=f"{label}<extra></extra>",
            )
        )
        start += count
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False, range=[-1, _COLS], fixedrange=True),
        yaxis=dict(visible=False, scaleanchor="x", range=[-25, 1], fixedrange=True),
        # Above the grid, so the key is read before the markers. The labels are too long for the side.
        legend=dict(orientation="v", yanchor="bottom", y=1.01, x=0, itemclick=False, itemdoubleclick=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
    )
    return fig
