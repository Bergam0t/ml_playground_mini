"""A pointer that shows a visitor what to click: each time they come to a tab (or a view of the
demo dashboard) with something clickable, a pointer glides onto it and clicks it, twice, with a ripple and a yellow flash of what it
clicked, then fades away. Used on the first tab's three boxes (it clicks "Missed most before")
and its patient buttons further down (it clicks "Medium risk", once they're scrolled into
view), and on the demo dashboard's All appointments table (it clicks the first row).

The pointer is drawn over the page, not in it, and lets every click through, so the page works
as normal throughout; any press removes the pointer at once. It waits until most of its target
is on screen, so a visitor who has to scroll down to it still sees it. If the script fails, the
page is just as it would be without it. Nothing moves for visitors who prefer reduced motion.

Streamlit's sanitiser drops a script whose text looks like it holds markup (a "<" then a letter
or "/"), so the script builds its pointer node by node.
"""

import json

import streamlit as st

CLICK_SCRIPT = """
<script>
(function () {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    // NAME: which pointer this is (several can be on one page).
    // TARGET: the keyed container to click; INNER: a selector for the element inside it to
    // click (a table, a button), or null for the container itself.
    // ROW: a table's row height, to click its first row; null to click the whole target.
    // DELAY: ms to wait once the target is on screen, so the page has settled.
    // BOTTOM: the fraction of the screen at the bottom that doesn't count as on screen, so a
    // target low on the page waits until the visitor has scrolled it well into view.
    var NAME = __NAME__, TARGET = __TARGET__, INNER = __INNER__, ROW = __ROW__;
    var DELAY = __DELAY__, BOTTOM = __BOTTOM__;
    var CYCLE = 1800;      // ms for one glide and click
    var CYCLES = 2;
    var WAIT = 60000;      // give up if the target is never drawn
    var YELLOW = '#ffe600';
    // A newer copy of this pointer (say, on a reloaded page) takes over from an older one
    var runs = window.__clickHintRuns = window.__clickHintRuns || {};
    var run = (runs[NAME] || 0) + 1;
    runs[NAME] = run;
    function current() { return window.__clickHintRuns[NAME] === run; }
    var started = performance.now();
    var overlay = null, grid = null, frame = null, watcher = null, observer = null, timer = null;

    function findGrid() {
        var box = document.querySelector('.st-key-' + TARGET);
        if (!box) return null;
        return INNER ? box.querySelector(INNER) : box;
    }

    function stop() {
        if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
        overlay = null;
        if (frame) cancelAnimationFrame(frame);
        if (watcher) watcher.disconnect();
        if (observer) observer.disconnect();
        if (timer) clearTimeout(timer);
        document.removeEventListener('pointerdown', stop, true);
        // Let another pointer waiting its turn play
        if (window.__clickHintPlaying === NAME + run) window.__clickHintPlaying = null;
    }

    function el(css) {
        var d = document.createElement('div');
        d.style.cssText = 'position:absolute;pointer-events:none;' + css;
        return d;
    }

    function play() {
        window.__clickHintPlaying = NAME + run;
        // The overlay sits over the target, on the page body so Streamlit's redraws leave it
        // alone, and follows the target each frame as the page scrolls
        overlay = el('position:fixed;z-index:999;overflow:visible;transition:opacity 300ms;');
        // What lights up on the click: a table's first row, under its header row, or the
        // whole target, with its own rounded corners
        var bar = el((ROW
            ? 'left:0;right:0;top:' + ROW + 'px;height:' + ROW + 'px;background:rgba(255,230,0,0.28);'
            : 'inset:0;border-radius:' + (getComputedStyle(grid).borderRadius || '0.5rem') +
              ';background:rgba(255,230,0,0.1);') +
            'box-shadow:inset 0 0 0 2px ' + YELLOW + ';opacity:0;');
        var ring = el('width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;' +
            'border:3px solid ' + YELLOW + ';opacity:0;');
        // A classic arrow pointer, its tip at the element's top-left corner
        var hand = el('width:34px;height:46px;transform-origin:0 0;opacity:0;' +
            'filter:drop-shadow(0 2px 4px rgba(0,0,0,0.6));');
        // Built node by node: Streamlit's sanitiser drops a script whose text holds markup
        var NS = 'http://www.w3.org/2000/svg';
        var svg = document.createElementNS(NS, 'svg');
        svg.setAttribute('width', '34');
        svg.setAttribute('height', '46');
        svg.setAttribute('viewBox', '0 0 18 27');
        var path = document.createElementNS(NS, 'path');
        path.setAttribute('d', 'M1 1 L1 22 L6.5 16.8 L10.6 25.6 L14.2 24 L10.2 15.4 L17.4 15.4 Z');
        path.setAttribute('fill', '#fff');
        path.setAttribute('stroke', '#111');
        path.setAttribute('stroke-width', '1.4');
        path.setAttribute('stroke-linejoin', 'round');
        svg.appendChild(path);
        hand.appendChild(svg);
        overlay.appendChild(bar);
        overlay.appendChild(ring);
        overlay.appendChild(hand);
        document.body.appendChild(overlay);

        function place() {
            if (!current() || !overlay) return stop();
            var r = grid.getBoundingClientRect();
            // Gone (the tab or view changed, or a table was redrawn empty), or scrolled right
            // off screen (so a pointer waiting its turn further down needn't wait): stop
            if (!grid.isConnected || r.width === 0 || r.height === 0 ||
                r.bottom <= 0 || r.top >= window.innerHeight) return stop();
            overlay.style.left = r.left + 'px';
            overlay.style.top = r.top + 'px';
            overlay.style.width = r.width + 'px';
            overlay.style.height = r.height + 'px';
            // The click lands a little way into a table's first row, or near a box's middle
            var x = ROW ? Math.min(r.width * 0.3, 240) : r.width * 0.5;
            var y = ROW ? ROW * 1.5 : r.height * 0.45;
            hand.style.left = ring.style.left = x + 'px';
            hand.style.top = ring.style.top = y + 'px';
            frame = requestAnimationFrame(place);
        }
        place();

        var timing = { duration: CYCLE, iterations: CYCLES, easing: 'linear', fill: 'both' };
        function at(dx, dy, s) { return 'translate(' + dx + 'px,' + dy + 'px) scale(' + s + ')'; }
        hand.animate([
            { offset: 0, transform: at(150, 110, 1), opacity: 0, easing: 'cubic-bezier(.2,.7,.2,1)' },
            { offset: 0.12, opacity: 1 },
            { offset: 0.45, transform: at(0, 0, 1), opacity: 1 },
            { offset: 0.52, transform: at(0, 0, 0.8), opacity: 1 },
            { offset: 0.6, transform: at(0, 0, 1), opacity: 1 },
            { offset: 0.85, transform: at(0, 0, 1), opacity: 1 },
            { offset: 1, transform: at(0, 0, 1), opacity: 0 }
        ], timing);
        ring.animate([
            { offset: 0, transform: 'scale(0.2)', opacity: 0 },
            { offset: 0.5, transform: 'scale(0.2)', opacity: 0 },
            { offset: 0.52, transform: 'scale(0.2)', opacity: 1, easing: 'ease-out' },
            { offset: 0.85, transform: 'scale(1.8)', opacity: 0 },
            { offset: 1, transform: 'scale(1.8)', opacity: 0 }
        ], timing);
        bar.animate([
            { offset: 0, opacity: 0 },
            { offset: 0.5, opacity: 0 },
            { offset: 0.56, opacity: 1 },
            { offset: 0.8, opacity: 1 },
            { offset: 1, opacity: 0 }
        ], timing);

        // Any press: the visitor is busy (or has got it), so stop at once
        document.addEventListener('pointerdown', stop, true);
        timer = setTimeout(stop, CYCLE * CYCLES + 100);
    }

    // Whether most of the target is on screen, above the bottom BOTTOM of it
    function inView() {
        var r = grid.getBoundingClientRect();
        var bottom = window.innerHeight * (1 - BOTTOM);
        var shown = Math.min(r.bottom, bottom) - Math.max(r.top, 0);
        return r.height > 0 && shown >= r.height * 0.6;
    }

    // After the pause: play, unless another pointer is playing (wait for it to finish, so two
    // never play at once) or the visitor has scrolled away meanwhile (wait for it again)
    function turn() {
        if (!current() || !grid.isConnected) return;
        if (window.__clickHintPlaying) { timer = setTimeout(turn, 400); return; }
        if (!inView()) { watch(); return; }
        play();
    }

    // Once most of the target is on screen, after a pause so the page has settled
    function watch() {
        observer = new IntersectionObserver(function (entries) {
            if (entries.some(function (e) { return e.intersectionRatio >= 0.6; })) {
                observer.disconnect();
                observer = null;
                timer = setTimeout(turn, DELAY);
            }
        }, { threshold: [0, 0.6, 1], rootMargin: '0px 0px -' + BOTTOM * 100 + '% 0px' });
        observer.observe(grid);
    }

    function tryStart() {
        if (!current()) return true;
        grid = findGrid();
        if (!grid || grid.getBoundingClientRect().height === 0) return false;
        watch();
        return true;
    }

    // The target may be drawn just after this script runs
    watcher = new MutationObserver(function () {
        if (tryStart()) { watcher.disconnect(); watcher = null; }
    });
    watcher.observe(document.body, { childList: true, subtree: true });
    (function poll() {
        if (!watcher) return;
        if (tryStart()) { watcher.disconnect(); watcher = null; return; }
        if (performance.now() - started < WAIT) requestAnimationFrame(poll);
        else { watcher.disconnect(); watcher = null; }
    })();
})();
</script>
"""


def replay_on_change(place):
    """Let every pointer play again once the visitor moves somewhere else: forget which have
    been shown when `place` (the open tab, and the demo dashboard's view) differs from the last
    run's. Reruns in the same place leave them shown. Call it once per run, before any
    show_click_hint."""
    if st.session_state.get("click_hint_place") != place:
        for key in [k for k in st.session_state if k.startswith("click_hint_seen_")]:
            del st.session_state[key]
        st.session_state["click_hint_place"] = place


def show_click_hint(
    name, target_key, row_height=None, delay=500, inner=None, bottom_margin=0
):
    """Play the pointer for `name` once each time the visitor comes to its tab or view (see
    replay_on_change): later calls with the same name there do nothing.
    It clicks the container keyed `target_key`, or the element inside it matching the CSS
    selector `inner` (a table, a button): the first row of a table whose rows (and header) are
    `row_height` px tall, or, with no row height, the whole target. It plays `delay` ms after
    most of the target is on screen, not counting the bottom `bottom_margin` of the screen
    (0.25: the target must be in the top three-quarters). Pointers on one page take turns.

    Call it only on a run where the target's tab is open: a tab's contents are drawn on every
    run, even behind another tab, and a pointer sent to a hidden tab is never seen but still
    counts as shown. The script's container (click_hint_*) is hidden by style.css."""
    seen = f"click_hint_seen_{name}"
    if st.session_state.get(seen):
        return
    st.session_state[seen] = True
    script = CLICK_SCRIPT
    for placeholder, value in {
        "__NAME__": name,
        "__TARGET__": target_key,
        "__INNER__": inner,
        "__ROW__": row_height,
        "__DELAY__": delay,
        "__BOTTOM__": bottom_margin,
    }.items():
        script = script.replace(placeholder, json.dumps(value))
    with st.container(key=f"click_hint_{name}"):
        st.html(script, unsafe_allow_javascript=True)
