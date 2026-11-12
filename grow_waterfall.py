"""The first tab's explanation chart builds itself up when a new patient is picked.

The rows play in the order the chart is read: the average appointment's bar grows from zero,
then each fact's bar sweeps out from where the one above ended (red to the right, blue to the
left), its row label and figure fading in with it, and last the patient's own total grows from
zero. A new patient's facts are different facts, so the old bars are not morphed into the new.

Streamlit draws the chart in its new state; the script below then plays it from empty. If the
script fails, the chart is still right, just without the motion. Nothing moves for visitors
who prefer reduced motion.
"""

import json

import streamlit as st

GROW_SCRIPT = """
<script>
(function () {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    var KINDS = __KINDS__, TOKEN = __TOKEN__, CHART = __CHART__;
    var BAR = 320;     // ms for one bar to sweep out
    var STEP = 110;    // ms between one fact's bar starting and the next
    var EASE = 'cubic-bezier(.2,.7,.2,1)';
    // A newer click takes over from an older run still waiting for its chart
    var run = (window.__waterfallRun || 0) + 1;
    window.__waterfallRun = run;
    var started = performance.now();
    var done = false;

    function findChart() {
        var gd = document.querySelector('.st-key-' + CHART + ' .js-plotly-plot');
        if (gd && gd.layout && gd.layout.meta === TOKEN &&
            gd.querySelectorAll('.waterfalllayer .points > .point').length === KINDS.length) return gd;
        return null;
    }

    // Each row's start: the average first, the facts one after another, the total last
    function startOf(i) {
        if (i === 0) return 0;
        if (KINDS[i] === 'total') return 180 + (KINDS.length - 2) * STEP + 120;
        return 180 + (i - 1) * STEP;
    }

    // The row labels fade in with their bars. Plotly redraws the axis just after drawing the
    // chart, which would drop an animation set on the labels themselves, so these are CSS
    // animations from a stylesheet, which a redrawn label picks up too. Named for this run, so
    // they play again each time; the sheet goes once they're done.
    function labels() {
        var sheet = document.getElementById('waterfall-grow-style');
        if (!sheet) {
            sheet = document.createElement('style');
            sheet.id = 'waterfall-grow-style';
            document.head.appendChild(sheet);
        }
        var name = 'waterfall-label-' + run;
        var rules = ['@keyframes ' + name + ' { from { opacity: 0; } to { opacity: 1; } }'];
        KINDS.forEach(function (_, i) {
            rules.push('.st-key-' + CHART + ' .yaxislayer-above > .ytick:nth-child(' + (i + 1) +
                ') { animation: ' + name + ' 220ms ' + startOf(i) + 'ms backwards; }');
        });
        sheet.textContent = rules.join('\\n');
        setTimeout(function () {
            if (window.__waterfallRun === run) sheet.textContent = '';
        }, startOf(KINDS.length - 1) + BAR + 600);
    }

    function animate(gd) {
        var points = gd.querySelectorAll('.waterfalllayer .points > .point');
        var lines = gd.querySelectorAll('.waterfalllayer .lines > .line');
        labels();
        KINDS.forEach(function (kind, i) {
            var start = startOf(i);
            var length = kind === 'total' ? BAR + 130 : BAR;
            var bar = points[i].querySelector('path');
            var figure = points[i].querySelector('text');
            [bar, figure, lines[i - 1]].forEach(function (el) {
                if (el) el.getAnimations().forEach(function (a) { a.cancel(); });
            });
            if (bar) {
                // From the bar's own start: its left end, or its right end for a blue bar,
                // which moves the risk down
                bar.style.transformBox = 'fill-box';
                bar.style.transformOrigin = (kind === 'down' ? 'right' : 'left') + ' center';
                bar.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }],
                    { duration: length, delay: start, easing: EASE, fill: 'backwards' });
            }
            var appear = [{ opacity: 0 }, { opacity: 1 }];
            if (figure) figure.animate(appear, { duration: 200, delay: start + length * 0.7, fill: 'backwards' });
            // The connector from the row above appears as this row sets off from its end
            if (i > 0 && lines[i - 1]) lines[i - 1].animate(appear, { duration: 120, delay: start, fill: 'backwards' });
        });
    }

    function tryStart() {
        if (done || window.__waterfallRun !== run) return true;
        var gd = findChart();
        if (!gd) return false;
        done = true;
        animate(gd);
        return true;
    }

    // A MutationObserver catches the new chart as it is drawn, before the browser paints it,
    // so it never shows complete first; the frame-by-frame check is a fallback
    var observer = new MutationObserver(function () { if (tryStart()) observer.disconnect(); });
    observer.observe(document.body, { childList: true, subtree: true });
    (function poll() {
        if (tryStart()) return observer.disconnect();
        if (performance.now() - started < 4000) requestAnimationFrame(poll);
        else observer.disconnect();
    })();
})();
</script>
"""


def row_kinds(expl):
    """What each row of charts.waterfall(expl) is, top to bottom: the average ("base"), each
    fact and the other factors ("up" or "down", the way it moves the risk) and the patient's
    total ("total")."""
    steps = list(expl["shown"]) + ([expl["other"]] if expl["other"] else [])
    return ["base", *("up" if s["value"] >= 0 else "down" for s in steps), "total"]


def grow_waterfall(expl, token, chart_key):
    """Play the waterfall keyed `chart_key` from empty, once its layout's meta is `token`. The
    script's container (intro_grow) is hidden by style.css."""
    script = (
        GROW_SCRIPT.replace("__KINDS__", json.dumps(row_kinds(expl)))
        .replace("__TOKEN__", json.dumps(token))
        .replace("__CHART__", json.dumps(chart_key))
    )
    with st.container(key="intro_grow"):
        st.html(script, unsafe_allow_javascript=True)
