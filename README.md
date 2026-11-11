# Outpatient DNA Forecast - demo build

Runs entirely on synthetic data. No SQL Server or .env file needed.

Based on https://github.com/WX-BIU/Outpatient-DNAs.

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What's in the app

- **Your rule vs the model** tab: a short game to show why a machine learning model can beat a person at
  this.
  - Built for a 2-5 minute visit to a stand: the setup is three numbers and a question, so the
    clue buttons are on screen without scrolling at 1920x1080 (100% and 125% scaling).
  - Using 1,000 past appointments (80 missed) and 100 calls, the visitor builds a rule for who
    to phone from the nine clues the model sees (including *riskiest times of day*), with each
    chosen clue getting an equal say.
    After each change, a callout says whether the last clue added or taken away made the list
    better or worse, and the rule is ranked against all 511 rules the buttons can make (the
    best is five clues together: 34 of 80).
  - They then watch a model learn its own weighting from 0 to 22,500 past appointments: the
    race against their best rule beside how much say each clue gets. A toast and an
    "Overtook you" marker show where the model passes them (800 for the best rule).
  - The verdict follows at once: a banner saying where the model overtook them, then two short
    cards: same clues but the model learnt which matter most (no rule from the buttons can match
    its 40), and even the best possible list (knowing each patient's true chance) reaches only
    44 of 80. A short "Why a computer?" section comes last.
  - **Start again** (at the top once a game has begun, and after the verdict) clears the game
    for the next visitor.
  - Two buttons lead on to the Upcoming appointments and How many people should we phone? tabs, as in the
    geographic mini. The tabs are keyed (`key="page"`) so a button can switch them.
- **Appointments** tab: a short introduction to the app, then the filterable appointment list, highest risk first.
  Click a row to see a SHAP waterfall explaining that appointment's predicted risk of not
  attending, with a "How to read this chart" guide. With no row selected, a bar chart shows
  what the model relies on most (mean |SHAP|) across all upcoming appointments, with a second
  chart beside it for the filtered appointments when the appointment-detail filters leave at
  least 30 (not shown for the DNA Risk and Confidence filters, which select by the forecast).
- **Weekly summary** tab: appointments and forecast DNAs by week and day.
- **How many people should we phone?** tab: choose who an admin team would phone, by the number of calls
  the team can make each week or by minimum predicted risk. A 1,000-marker chart
  (ported from lambda_ml_streamlit_exercise) shows, for past appointments, how many calls would
  reach someone who then did not attend, and how many such patients would get no call. An
  assumed-effect slider shows how many missed appointments the calls might avoid. We don't know
  that a call changes whether someone attends, and the tab says so. A "What might the calls cost,
  and save?" section turns the weekly figures into staff hours, the cost of the calls, the value
  of the missed appointments they reach and might avoid, and the **break-even effect**: the share
  of people reached a call would need to get to attend to cover its own staff time.

Both views also show the expected cost of missed appointments
(each predicted risk times that appointment's cost).

The sidebar filters apply to the Upcoming appointments tab (there is no date filter: click a day in the weekly summary instead). How many people should we phone?
hides them and shows only the cost assumptions: its calls-per-week figures use all the upcoming
appointments, and the chart uses past appointments. The filters keep their settings for the other
tabs (the script checks the open tab, which it can since the tabs are keyed). Your rule vs the
model uses past appointments only, so the sidebar is hidden while that tab is open (a CSS rule
in `style.css` keyed on the selected tab, so the filters keep their settings for the other tabs).

## Screen size

The page names in the top bar show as four buttons when there is room. When the bar is narrower
than about 1400px, they become a dropdown instead. The sidebar is Streamlit's default 300px
(narrower than the other minis', to leave the page more room), so with it open the browser window
needs to be about 1775px wide for the buttons (about 1475px with it shut).

Windows display scaling reduces the width the browser sees. On a 1920x1080 screen:

| Scaling | Width the browser sees | Top bar |
|---|---|---|
| 100% | 1920px | the buttons |
| 125% | 1536px | the buttons with the sidebar shut, the dropdown with it open |
| 150% | 1280px | the dropdown |

To get the buttons back, zoom the browser out (Ctrl and minus). Everything still works with the
dropdown; it just takes one more click to change page.

Below 1300px wide, some side-by-side rows wrap onto more lines (the keyed `reflow_*` containers in
`streamlit_app.py`, styled at the end of `style.css`). On phones the sidebar opens over the page and
can be shut and reopened with the arrows at the top left.

Streamlit is pinned to 1.66.1 in `requirements.txt`, because `style.css` and `mobile_nav.py` depend on
Streamlit's internal markup (`stToolbar`, `data-collapsed`, `stLayoutWrapper`, the tab bar), which can
change in any release. After upgrading, re-check the sidebar's reopen button, the top bar and the page
dropdown at desktop and phone widths.

## Cost assumptions

The sidebar's **Cost assumptions** section holds editable defaults, set in `costs.py`:

- **Calls:** a band 5 admin at about £26 an hour. That's the Agenda for Change 2025/26 middle
  pay point (£33,487) plus employer National Insurance and pension, over about 1,625 productive
  hours a year. At 3 minutes a call, a call costs about £1.31.
- **Missed appointments:** a cost by specialty and New / Follow Up, from £100 to £260
  (about £140 on average for the current mix). These are illustrative values, loosely in line
  with NHS Payment Scheme outpatient prices: replace them with your Trust's own costs.
- **Share of the cost really lost:** 100% by default. Lower it if empty slots get re-used.

A missed appointment's cost is the value of a clinic slot, not cash saved. The call cost counts
staff time only.

## Where the data and explanations come from

The app reads precomputed files in `data/`: these four, plus the two for the Your rule vs the
model tab (described below).

- `forecast.csv`: the future appointments with the model's predicted DNA risk
- `shap_values.csv`: each appointment's SHAP contributions, in percentage points
- `past_predictions.csv`: the model's risk and confidence for the past appointments it was
  tested on (not trained on), with whether each patient really attended
- `model_meta.json`: the average (base) risk, test AUC and package versions

They are made offline by `build/train_and_explain.py`, which follows the pipeline in
lambda_ml_streamlit_exercise:

1. `demo_data.generate_appointments()` makes up past appointments with a hidden "true" chance of not attending,
   and draws whether each patient attended. The chances average `DNA_RATE` (8%: recent NHS
   outpatient figures are about 6.4%, so the demo Trust is a little worse than average).
2. An XGBoost model is trained on those past appointments.
3. The model scores 3,200 future appointments. Their confidence score is worked out as in the
   live pipeline (WX-BIU/Outpatient-DNAs, `src/forecast/Outpatient_Forecast.py`): the top class
   probability minus the second, which for a yes/no model is |2q − 1|. The live model's q is
   inflated about four times by `scale_pos_weight=4`, so its 50% is about a 20% true risk. The demo's
   model is calibrated, so the risk is first inflated the same way (odds × 4). Confidence is then 0
   at the 20% forecast line and goes up with the risk for every appointment forecast as a DNA.
4. `shap.TreeExplainer` explains each prediction on the probability scale.

**Patient name and appointment time.** Each patient has a made-up name (the same on all their
appointments), and each appointment a clinic time from 08:30 to 16:45 in 15-minute steps (fewer over
lunch). Both come from a separate random stream, so adding them left every other column unchanged. The
name is shown only on the call list (Show next week call list), with the appointment time. The model sees
the hour (`Time_Slot`, 08:00 to 16:00) as a clue: appointments starting 10-12 or 15-17 miss more
(`TIME_EFFECT` in `demo_data.py`), and Monday before midday gets an extra push
(`MONDAY_MORNING_EFFECT`) on top of Monday's own, so it is the worst slot of the week (about 19%
missed, against 6% for the rest of the week). The build checks this. Explanations show the real time
(`Appointment_Time`), not the hour slot.

Dates are not stored. `demo_data.load_forecast()` dates the appointments from next Monday onwards,
so they always roll forward from today while each appointment keeps its weekday.

To rebuild (for example after changing `SEED` in `build/train_and_explain.py` or the effects in
`demo_data.py`):

```
pip install -r build/requirements-build.txt
python build/train_and_explain.py
python build/rule_game_data.py
```

`build/rule_game_data.py` makes the two files for the Your rule vs the model tab. It uses the same
split as the main build, and checks that its test appointments match `past_predictions.csv`.

- `rule_game_past.csv`: the past test appointments with a "phone first" score per clue, as a
  sensible person would use it. Numbers point the way that goes with more missed appointments,
  and categories are scored by last year's miss rate.
- `rule_game_learning.json`: the model's results after learning from 0, 25, 50, ... 22,500 past
  appointments.

The model on this tab is a logistic regression on raw inputs, so it has to learn which way each clue
points as well as how much it counts. That's why it starts out worse than a person. With all the
data it reaches the same share of misses as the XGBoost model (the build checks this). The demo
data has only one interaction between clues, Monday mornings (below), which the simpler model can't
learn, but it is a small part of the risk, so the simpler model loses almost nothing.

The build checks that the SHAP values add up to each prediction, that the past DNA rate is
within a point of `DNA_RATE` and that every risk band has some appointments. It prints how the
forecasts fall into the risk bands and the high-risk flag.

It also checks **calibration** on the past appointments the model didn't train on: in each risk
band, the average predicted risk must be close to the share who really missed. The app adds risks
up (expected missed appointments, costs), so they must be true probabilities. That's why the
demo model isn't class-weighted. The live model trains with `scale_pos_weight=4`, which makes its
risks roughly four times the true odds, and that is why its thresholds (15/50/80% bands, a 70%
flag) sit so much higher than the demo's.

## Risk bands and the high-risk flag

The live app's thresholds were set for much higher risks than an 8% DNA rate produces (its
high-risk flag, risk over 70% and confidence over 70%, would catch almost nobody). The demo uses:

- **Risk bands** (`DNA_Risk()` in `streamlit_app.py`): Low under 5%, Medium 5% to 20%, High 20% to
  40%, Very High 40% or more.
- **Yes/no forecast** ("Expected DNA", `FORECAST_MIN` in the build and the app): a predicted risk
  of 20% or more, about two and a half times the average. The live model says yes at 50%, which
  with its `scale_pos_weight=4` is about a 20% true risk. That's about 9% of appointments; about
  40% of those past patients missed, and they include about half of everyone who missed.
- **High-risk flag** (`HIGH_RISK_MIN`): the Very High band, 40% or more, so it stays stricter than
  the forecast as in the live app. About 3% of appointments; about 58% of those past patients
  missed.
- **Weekly summary colours** (`weekly.MEASURES`): green below and red above 2% and 6% for the
  flag, 5% and 15% for the yes/no forecast, and 5% and 10% for the average predicted risk.

To go back to live data, replace `load_forecast()` in `demo_data.py` with a read of the combined
forecast view. You would also need SHAP values from the live model for the waterfalls.
