"""Markers that move between the boxes of the first tab's "Are we calling the right people?"
chart when a different tile is clicked.

All three ways make the same number of calls, so the column totals never change: only the
split inside each column does. If the new way's calls reach d more people who would have
missed, d of them move from "Missed, and got no call" to "Useful call" (a red cross becomes a
green dot), and d people who came move the other way, from "Unnecessary call" to "Came, and got
no call". These are the net moves, not every patient who changes list.

The chart is redrawn in its new state by Streamlit; the script below then hides the arriving
markers, flies a copy of each in from where it was on the old chart, changing shape on the way,
and shows the real marker again when its copy lands. If the script fails, the chart is still
right, just without the motion. Nothing moves for visitors who prefer reduced motion.
"""

import json

import streamlit as st

import charts

# The four boxes in the order outcome_blocks() draws their marker traces
TRACE_ORDER = ["called_dna", "called_attended", "not_called_dna", "not_called_attended"]

# outcome_blocks()' plotly marker symbols, as the shapes the script draws
_SHAPES = {"circle": "dot", "square": "square", "x-thin-open": "cross", "circle-open": "ring"}

MOVE_SCRIPT = """
<script>
(function () {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    var MOVERS = __MOVERS__, STYLES = __STYLES__, TOKEN = __TOKEN__, CHART = __CHART__;
    var SIZE = __SIZE__, LINE = __LINE__, TRACES = __TRACES__;
    var NS = 'http://www.w3.org/2000/svg';
    var FLIGHT = 900;                                   // ms per marker
    var STAGGER = Math.min(30, 700 / Math.max(1, MOVERS.length / 2));
    // A newer click takes over: an older run stops waiting and leaves the points alone
    var run = (window.__markerMoveRun || 0) + 1;
    window.__markerMoveRun = run;
    var started = performance.now();
    // and shows anything the older run had hidden, in case this one never finds its chart
    var leftover = document.getElementById('marker-move-style');
    if (leftover) leftover.textContent = '';
    document.querySelectorAll('svg.marker-move').forEach(function (old) { old.remove(); });

    // The new chart: Streamlit may draw it before or after this script runs
    function findChart() {
        var gd = document.querySelector('.st-key-' + CHART + ' .js-plotly-plot');
        if (gd && gd._fullLayout && gd.layout && gd.layout.meta === TOKEN &&
            gd.querySelectorAll('.scatterlayer .trace').length === TRACES.length) return gd;
        return null;
    }

    function wait() {
        if (window.__markerMoveRun !== run) return;
        var gd = findChart();
        if (gd) return animate(gd);
        if (performance.now() - started < 4000) requestAnimationFrame(wait);
    }

    function shape(kind, colour) {
        var r = SIZE / 2, el;
        if (kind === 'square') {
            el = document.createElementNS(NS, 'rect');
            el.setAttribute('x', -r); el.setAttribute('y', -r);
            el.setAttribute('width', SIZE); el.setAttribute('height', SIZE);
            el.setAttribute('fill', colour);
        } else if (kind === 'cross') {
            el = document.createElementNS(NS, 'path');
            el.setAttribute('d', 'M' + -r + ',' + -r + 'L' + r + ',' + r + 'M' + r + ',' + -r + 'L' + -r + ',' + r);
            el.setAttribute('fill', 'none');
        } else {
            el = document.createElementNS(NS, 'circle');
            el.setAttribute('r', r);
            el.setAttribute('fill', kind === 'dot' ? colour : 'none');
        }
        el.setAttribute('stroke', colour);
        el.setAttribute('stroke-width', LINE);
        return el;
    }

    function animate(gd) {
        var fl = gd._fullLayout, xa = fl.xaxis, ya = fl.yaxis;
        function px(x) { return xa._offset + (x - xa.range[0]) / (xa.range[1] - xa.range[0]) * xa._length; }
        function py(y) { return ya._offset + (ya.range[1] - y) / (ya.range[1] - ya.range[0]) * ya._length; }
        // Hide the markers that will arrive, until their copies land. They are the last ones in
        // their box, so one rule per box hides them all from the next to land onwards. A
        // stylesheet rather than the points' own style, as plotly may redraw the points;
        // !important, as plotly sets each point's opacity in its style attribute.
        var nextToLand = {};
        MOVERS.forEach(function (m) {
            nextToLand[m.d] = Math.min(m.i, nextToLand[m.d] === undefined ? m.i : nextToLand[m.d]);
        });
        var sheet = document.getElementById('marker-move-style');
        if (!sheet) {
            sheet = document.createElement('style');
            sheet.id = 'marker-move-style';
            document.head.appendChild(sheet);
        }
        function hide() {
            sheet.textContent = Object.keys(nextToLand).map(function (key) {
                return '.st-key-' + CHART + ' .scatterlayer > .trace:nth-child(' +
                    (TRACES.indexOf(key) + 1) + ') .points > path:nth-child(n+' +
                    (nextToLand[key] + 1) + ') { opacity: 0 !important; }';
            }).join('\\n');
        }
        hide();

        var host = gd.querySelector('.svg-container') || gd;
        host.querySelectorAll('svg.marker-move').forEach(function (old) { old.remove(); });
        var layer = document.createElementNS(NS, 'svg');
        layer.setAttribute('class', 'marker-move');
        layer.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;' +
            'overflow:visible;pointer-events:none;';
        host.appendChild(layer);

        var left = MOVERS.length;
        MOVERS.forEach(function (m) {
            var fx = px(m.f[0]), fy = py(m.f[1]), tx = px(m.t[0]), ty = py(m.t[1]);
            // A low arc, so the markers don't cross the boxes' titles
            var lift = Math.min(14, 4 + Math.abs(tx - fx) * 0.04);
            var outer = document.createElementNS(NS, 'g');   // along the line
            var inner = document.createElementNS(NS, 'g');   // the arc above it
            var from = shape(STYLES[m.s].shape, STYLES[m.s].colour);
            var to = shape(STYLES[m.d].shape, STYLES[m.d].colour);
            to.style.opacity = 0;
            inner.appendChild(from); inner.appendChild(to);
            outer.appendChild(inner); layer.appendChild(outer);
            outer.style.transform = 'translate(' + fx + 'px,' + fy + 'px)';

            var timing = { duration: FLIGHT, delay: m.k * STAGGER, fill: 'both' };
            outer.animate([
                { transform: 'translate(' + fx + 'px,' + fy + 'px)' },
                { transform: 'translate(' + tx + 'px,' + ty + 'px)' }
            ], Object.assign({ easing: 'cubic-bezier(.5,0,.3,1)' }, timing));
            inner.animate([
                { transform: 'translateY(0)', easing: 'ease-out' },
                { transform: 'translateY(' + -lift + 'px) scale(1.35)', easing: 'ease-in' },
                { transform: 'translateY(0)' }
            ], timing);
            from.animate([{ opacity: 1 }, { opacity: 1, offset: 0.3 }, { opacity: 0, offset: 0.7 }, { opacity: 0 }], timing);
            var landing = to.animate([{ opacity: 0 }, { opacity: 0, offset: 0.3 }, { opacity: 1, offset: 0.7 }, { opacity: 1 }], timing);
            landing.onfinish = function () {
                if (window.__markerMoveRun !== run) return;
                // Each box's copies land in order, so show its markers up to this one
                nextToLand[m.d] = Math.max(nextToLand[m.d], m.i + 1);
                hide();
                outer.remove();
                if (--left === 0) {
                    layer.remove();
                    sheet.textContent = '';
                }
            };
        });
    }

    wait();
})();
</script>
"""


def outcome_moves(old, new, left_cols, right_cols):
    """The markers that move from the `old` outcomes (calls.call_outcomes) to the `new`, laid out
    as outcome_blocks(..., left_cols, right_cols, compact=True) draws them. Each is a dict:
    f and t, where it starts and lands in data units; s and d, the boxes it leaves and joins;
    i, its index among the markers of box d; k, its turn to set off. Both calls must make the
    same number of calls, so that the column totals stay the same."""
    gained = new["called_dna"] - old["called_dna"]
    n = abs(gained)
    if not n or old["calls"] != new["calls"]:
        return []
    before = charts.outcome_positions(old, left_cols, right_cols, compact=True)
    after = charts.outcome_positions(new, left_cols, right_cols, compact=True)
    if gained > 0:
        pairs = [("not_called_dna", "called_dna"), ("called_attended", "not_called_attended")]
    else:
        pairs = [("called_dna", "not_called_dna"), ("not_called_attended", "called_attended")]
    moves = []
    for src, dst in pairs:
        # The last marker leaves first, and the first new place fills first, so they pour
        # from the end of one grid onto the end of the other
        for k in range(n):
            start = before[src][old[src] - 1 - k]
            end_i = new[dst] - n + k
            end = after[dst][end_i]
            moves.append(dict(
                f=[round(start[0], 2), round(start[1], 2)],
                t=[round(end[0], 2), round(end[1], 2)],
                s=src, d=dst, i=end_i, k=k,
            ))
    return moves


def animate_moves(moves, token, chart_key):
    """Fly `moves` (outcome_moves()) on the plotly chart keyed `chart_key`, once its layout's
    meta is `token`. The script's container (landing_move) is hidden by style.css."""
    if not moves:
        return
    size = charts._SIZES[True]
    styles = {
        key: {"colour": colour, "shape": _SHAPES[symbol]}
        for key, (colour, symbol) in charts._ICON_STYLE.items()
    }
    script = (
        MOVE_SCRIPT.replace("__MOVERS__", json.dumps(moves))
        .replace("__STYLES__", json.dumps(styles))
        .replace("__TOKEN__", json.dumps(token))
        .replace("__CHART__", json.dumps(chart_key))
        .replace("__SIZE__", json.dumps(size["marker"]))
        .replace("__LINE__", json.dumps(size["line"]))
        .replace("__TRACES__", json.dumps(TRACE_ORDER))
    )
    with st.container(key="landing_move"):
        st.html(script, unsafe_allow_javascript=True)
