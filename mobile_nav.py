"""Page dropdown that replaces the navbar's tab pills on narrower screens.

Ported from the geographic playground mini (which took it from the DES playground mini).
style.css shows it where the title and three pills no longer fit side by side.
"""

import json

import streamlit as st

# Adds a <select> to the top-level tab bar, mirroring its tabs. style.css only
# shows it when the bar is under about 1100px wide, where it replaces the tab pills.
# Choosing an option clicks the matching tab; a MutationObserver keeps the
# selected option in step when the tab changes, and adds the select back if
# Streamlit redraws the bar.
MOBILE_NAV_SCRIPT = """
<script>
(function () {
    if (window.__mobileNavInstalled) return;
    window.__mobileNavInstalled = true;

    // Shorter names for tabs whose full label doesn't fit beside the title
    var shortLabels = __SHORT_LABELS__;

    function topTabList() {
        return Array.from(document.querySelectorAll('[role="tablist"]')).find(
            function (list) { return !list.closest('[data-testid="stTabPanel"]'); }
        );
    }

    function tabsOf(list) {
        return Array.from(list.querySelectorAll('[data-testid="stTab"]'));
    }

    function sync() {
        var list = topTabList();
        if (!list) return;
        var tabs = tabsOf(list);
        if (!tabs.length) return;

        var select = list.querySelector('select.mobile-nav');
        if (!select) {
            select = document.createElement('select');
            select.className = 'mobile-nav';
            select.setAttribute('aria-label', 'Page');
            select.addEventListener('change', function () {
                var tab = tabsOf(topTabList())[select.selectedIndex];
                if (tab) tab.click();
            });
            list.appendChild(select);
        }

        var labels = tabs.map(function (tab) {
            var label = tab.textContent.trim();
            return shortLabels[label] || label;
        });
        var current = Array.from(select.options).map(function (o) { return o.text; });
        if (labels.join('\\n') !== current.join('\\n')) {
            select.replaceChildren.apply(select, labels.map(function (label) {
                var option = document.createElement('option');
                option.text = label;
                return option;
            }));
        }

        var selected = tabs.findIndex(function (tab) {
            return tab.getAttribute('data-selected') === 'true';
        });
        if (selected >= 0 && select.selectedIndex !== selected) {
            select.selectedIndex = selected;
        }
    }

    var queued = false;
    new MutationObserver(function () {
        if (queued) return;
        queued = true;
        requestAnimationFrame(function () { queued = false; sync(); });
    }).observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-selected'],
    });
    sync();
})();
</script>
"""


def add_mobile_nav(short_labels=None):
    """
    Add a dropdown that replaces the navbar's tab pills when the bar is under about 1100px wide.

    Runs ``MOBILE_NAV_SCRIPT`` on the page. The script's own container is hidden
    (``.st-key-mobile_nav`` in style.css) so it takes up no space; the script
    still runs.

    Params:
    ------
    short_labels: dict or None
        Maps a tab's full label to the shorter name shown in the dropdown, for
        labels too long to fit beside the title on a phone.
    """
    script = MOBILE_NAV_SCRIPT.replace(
        "__SHORT_LABELS__", json.dumps(short_labels or {})
    )
    with st.container(key="mobile_nav"):
        st.html(script, unsafe_allow_javascript=True)
