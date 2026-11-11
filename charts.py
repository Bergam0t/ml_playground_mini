"""Plotly charts for the SHAP explanations, styled for the app's dark theme.

waterfall() is ported from lambda_ml_streamlit_exercise/app/charts.py. guide_waterfall()
follows guide_chart() in geographic_playground_mini/pareto.py: a small made-up example
with numbered badges that match the steps written beside it. outcome_blocks() is adapted from
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


# Made-up example for the "How to read this chart" box: average 8%, three factors, 27%
GUIDE_STEPS = [
    ("Average appointment", 8, "absolute", "8%"),
    ("Missed appointments = 3", 18, "relative", "+18"),
    ("Days since booking = 60", 4, "relative", "+4"),
    ("Age = 70", -3, "relative", "-3"),
    ("This appointment", 27, "total", "27%"),
]
# Which row each numbered badge sits beside (the numbers match the steps written next to it)
GUIDE_BADGES = {1: "Average appointment", 2: "Missed appointments = 3", 3: "This appointment"}


def guide_waterfall() -> go.Figure:
    """Small annotated example waterfall with numbered badges in a column on the right."""
    labels, values, measure, texts = (list(t) for t in zip(*GUIDE_STEPS))
    fig = go.Figure(_waterfall_trace(labels, values, measure, texts))
    badge_x = 38
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


YOU = FILTERED  # the visitor's rule on the "Your rule vs the model" tab
MODEL = TOTAL  # the model learning the points
GRID = "rgba(255,255,255,0.1)"


def say_bars(order: list, labels: dict, model_say: dict, your_clues) -> go.Figure:
    """How much say each clue gets: an equal share for each clue in the visitor's rule, and
    what the model has learnt. `order` is fixed by the caller so bars don't jump between
    animation frames."""
    yours = {f: (1 / len(your_clues) if f in your_clues else 0) for f in order}
    fig = go.Figure()
    for name, say, colour in (("Your currently selected rule", yours, YOU), ("The model", model_say, MODEL)):
        fig.add_trace(
            go.Bar(
                orientation="h",
                name=name,
                y=[labels[f] for f in order],
                x=[say[f] for f in order],
                marker=dict(color=colour),
                text=[f"{say[f]:.0%}" if say[f] >= 0.005 else "" for f in order],
                textposition="outside",
                textfont=dict(color=TEXT, size=13),
                cliponaxis=False,
                hovertemplate=f"{name}: %{{y}} has %{{x:.0%}} of the say<extra></extra>",
            )
        )
    _layout(
        fig,
        110 + 46 * len(order),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, itemclick=False, itemdoubleclick=False),
        margin=dict(l=0, r=40, t=30, b=10),
        xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor=GRID, zeroline=False, fixedrange=True),
    )
    # _layout hides the legend; this chart needs it to tell the two bars apart
    fig.update_layout(showlegend=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def learning_curve(
    frames: list,
    upto: int,
    your_best: int | None,
    perfect: int,
    n_missed: int,
    overtook: int | None = None,
    height: int = 330,
) -> go.Figure:
    """Missed appointments (per 1,000 past appointments) on the model's call list, by how
    many past appointments it has learnt from, drawn up to frame `upto`. The x axis is the
    full set of sizes from the start (evenly spaced, as each roughly doubles), so it doesn't
    move while it draws. `your_best` and `perfect` are counts on the same scale. `overtook`
    is the frame where the model first beats your best: marked once the line reaches it. The
    newest point is labelled with the model's score."""
    sizes = [f"{f['n']:,}" for f in frames]
    shown = frames[: upto + 1]
    fig = go.Figure(
        go.Scatter(
            x=sizes[: upto + 1],
            y=[f["outcomes"]["called_dna"] for f in shown],
            mode="lines+markers",
            line=dict(color=MODEL, width=3),
            marker=dict(size=9, color=MODEL),
            name="The model",
            hovertemplate=f"After %{{x}} past appointments: %{{y}} of the {n_missed} missed appointments correctly predicted<extra></extra>",
        )
    )
    lines = [(perfect, "Best possible list", MUTED, "dot")]
    if your_best is not None:
        lines.append((your_best, "Your best rule", YOU, "dash"))
    for value, label, colour, dash in lines:
        fig.add_hline(y=value, line=dict(color=colour, dash=dash, width=2))
        fig.add_annotation(
            x=0, xref="paper", xanchor="left", y=value, yanchor="bottom",
            text=f"{label}: {value} of {n_missed}", showarrow=False, font=dict(color=colour, size=13),
        )
    # The model's latest score, beside its newest point: to the right while there is room, under
    # it for the last two points (the line is flat there, so nothing is in the way)
    latest = shown[-1]["outcomes"]["called_dna"]
    near_end = upto >= len(sizes) - 2
    fig.add_annotation(
        x=upto, y=latest, text=f"<b>Computer: {latest} of {n_missed}</b>", showarrow=False,
        xanchor="right" if near_end else "left", yanchor="top" if near_end else "middle",
        xshift=0 if near_end else 12, yshift=-12 if near_end else 0,
        font=dict(color=MODEL, size=13),
    )
    if overtook is not None and overtook <= upto:
        fig.add_annotation(
            # By position on the category axis: plotly reads a label such as "400" as a number
            x=overtook, y=frames[overtook]["outcomes"]["called_dna"],
            text="<b>Overtook you</b>", showarrow=True, arrowhead=2, arrowcolor="#ffe600",
            ax=-60, ay=-45, font=dict(color="#1b1a1a", size=14), bgcolor="#ffe600", borderpad=4,
        )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        showlegend=False,
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(
            title="Past appointments learnt from",
            type="category", categoryorder="array", categoryarray=sizes,
            range=[-0.5, len(sizes) - 0.5], gridcolor=GRID, fixedrange=True,
        ),
        yaxis=dict(
            title=f"Correctly predicted (of {n_missed})", range=[0, n_missed * 0.65],
            gridcolor=GRID, zeroline=False, fixedrange=True,
        ),
    )
    return fig


# The dataviz skill's fixed status steps (good / warning / critical), plus a recessive
# grey for the large "nothing happens" group. Green and red are close for deuteranopes,
# so every group also has its own marker shape, a label with its count and a hover label:
# colour is never the only cue.
_ICON_STYLE = {
    "called_dna": ("#0ca30c", "circle"),
    "called_attended": ("#fab219", "square"),
    "not_called_dna": ("#d03b3b", "x-thin-open"),
    "not_called_attended": (MUTED, "circle-open"),
}

# outcome_blocks(): four tinted boxes, two per column. Left column = on the call list, right =
# not on it. Each box says in plain words what its group means, so "attended" on the call list
# reads as an unnecessary call. Title and explanation per group (the hover keeps the GROUPS label).
_OUTCOME_TEXT = {
    "called_dna": ("●", "Useful call", "They would have missed<br>without it"),
    "called_attended": ("■", "Unnecessary call", "They would have attended<br>anyway"),
    "not_called_dna": ("✕", "Missed, and got no call", "We never reached them"),
    "not_called_attended": ("○", "Came, and got no call", "No call needed"),
}
_COLUMNS = [
    ("On the call list", ["called_dna", "called_attended"]),
    ("Not on the list", ["not_called_dna", "not_called_attended"]),
]
# Sizes in pixels. unit: one marker row (and column). A box is at least min_box wide for its
# text, and reserves box_head for its heading (title + explanation) and col_head for the
# column heading above. The compact set is for the phone tab, where the chart sits beside the
# cards and above the fold: smaller markers and text, and each explanation on one line.
_SIZES = {
    False: dict(unit=16, marker=9, line=2, col_font=22, sym_font=18, title_font=15, why_font=13,
                col_head=44.8, box_head=80, min_box=256),
    True: dict(unit=12, marker=7, line=1.5, col_font=18, sym_font=15, title_font=14, why_font=12,
               col_head=30, box_head=48, min_box=230),
}
_HEADING_H = 2.0  # the optional line above the column headings, in units


def _tint(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def outcome_blocks(
    outcomes: dict,
    left_cols: int,
    right_cols: int,
    heading: str = "",
    compact: bool = False,
) -> go.Figure:
    """The 1,000 past appointments as four boxes, two per column: those on the call list (left)
    and those not (right), each split into who would have missed and who would have attended.
    Every box has its own heading and plain-words explanation in place of a legend. Markers are
    laid out left_cols / right_cols wide, so the Your rule tab can keep its 100 calls in a 10 x 10
    square and the phone tab can widen the columns with the number of calls. heading, if given,
    runs across the top of both columns to say how the list was made. compact draws it smaller
    and shorter (see _SIZES); give it about 85 columns in all to fill a 1,030px-wide column."""
    size = _SIZES[compact]
    unit = size["unit"]
    min_box_w, box_head, col_head = (size[k] / unit for k in ("min_box", "box_head", "col_head"))
    labels = dict(GROUPS)
    fig = go.Figure()
    x0 = 0.0
    total_h = 0.0
    top = _HEADING_H if heading else 0.0
    if heading:
        fig.add_annotation(
            x=-0.4, y=0, xref="x", yref="y", xanchor="left", yanchor="top", align="left",
            text=f"<b>{heading}</b>", showarrow=False, font=dict(size=16, color=MUTED),
        )
    for (title, keys), cols in zip(_COLUMNS, (left_cols, right_cols)):
        cols = max(1, cols)
        box_w = max(cols, min_box_w)
        n_col = sum(outcomes[k] for k in keys)
        fig.add_annotation(
            x=x0 - 0.4, y=-top, xref="x", yref="y", xanchor="left", yanchor="top", align="left",
            text=f"<b>{title}: {n_col}</b>", showarrow=False, font=dict(size=size["col_font"], color=TEXT),
        )
        y = -top - col_head
        for key in keys:
            colour, symbol = _ICON_STYLE[key]
            sym, name, why = _OUTCOME_TEXT[key]
            if compact:
                why = why.replace("<br>", " ")
            count = outcomes[key]
            rows = -(-count // cols) if count else 0
            box_h = box_head + rows + 0.6
            fig.add_shape(
                type="rect", x0=x0 - 0.5, x1=x0 - 0.5 + box_w + 0.4, y0=y, y1=y - box_h,
                fillcolor=_tint(colour, 0.10), line=dict(color=_tint(colour, 0.6), width=1),
                layer="below",
            )
            fig.add_annotation(
                x=x0 - 0.2, y=y - 0.3, xref="x", yref="y", xanchor="left", yanchor="top", align="left",
                text=(
                    f"<span style='color:{colour};font-size:{size['sym_font']}px'>{sym}</span> "
                    f"<b>{count} · {name}</b><br>"
                    f"<span style='font-size:{size['why_font']}px;color:{TEXT}'>{why}</span>"
                ),
                showarrow=False, font=dict(size=size["title_font"], color=TEXT),
            )
            idx = range(count)
            fig.add_trace(
                go.Scatter(
                    x=[x0 + i % cols for i in idx],
                    y=[y - box_head - (i // cols) - 0.3 for i in idx],
                    mode="markers",
                    marker=dict(color=colour, symbol=symbol, size=size["marker"],
                                line=dict(width=size["line"], color=colour)),
                    hovertemplate=f"{labels[key]}<extra></extra>",
                )
            )
            y -= box_h + 0.8
        total_h = max(total_h, -y)
        x0 += box_w + 1.2
    fig.update_layout(
        height=round((total_h + 0.5) * unit),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.6, x0 - 0.8], fixedrange=True,
                   constrain="domain", constraintoward="left"),
        yaxis=dict(visible=False, scaleanchor="x", range=[-total_h - 0.3, 0.4], fixedrange=True,
                   constrain="domain", constraintoward="top"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
    )
    return fig
