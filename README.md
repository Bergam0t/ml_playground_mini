# Outpatient DNA Forecast - demo build

Runs entirely on synthetic data. No SQL Server or .env file needed.

Based on https://github.com/WX-BIU/Outpatient-DNAs.

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What's in the app

The tabs are in the order a senior visitor with a couple of minutes needs them: why the model
helps and how it reaches a risk first, then how many calls to make, then a game as a bonus for those
who stay, and last a demo of the real tool's dashboard for those who want to try it.

- **Why use a model?** tab (the landing page). Uses all the upcoming appointments: the sidebar shows
  only the cost assumptions here, not the filters.
  - A short introduction beside the demonstration notice (which gives the real pilot's result).
  - **Why not just phone those who've missed before?**: of 1,000 past appointments (79
    missed), how many would 100 calls have reached who then missed, phoning the model's list
    (41), phoning those who missed most before (23) and at random (8, a quieter tile, as the
    baseline). The question sits in bold right above the tiles, and each tile reads as a sentence
    ("41 / of 100 calls / reached someone who would have missed") so it makes sense on its own.
    It leads with what the simple rule can't do: 6 in 10 missed appointments are by people who'd
    never missed before. A yellow box (like the demonstration notice) says the appointments are
    synthetic: the factors reflect what the real model found (its common reasons), but not
    exactly, as how much each one counts is made up (the bonus game's demonstration caption says
    the same about its clues). Worked out by `rule_game.three_ways()` from `rule_game_past.csv`
    and `past_predictions.csv` (the same rows). Each tile also shows the same in £ over the
    upcoming weeks: the value of appointments kept by phoning every forecast DNA (the model's
    tile), or the same number of patients chosen by the simple rule or at random (valued the same
    way). A note under the tiles says what the £ is and gives the cost of the calls once, as it's
    the same all three ways (clinic time kept, not cash).
  - **Each tile is clickable** (an invisible button stretched over it by `style.css`; the chosen
    tile is outlined in teal). Under the tiles, the phone tab's **Are we calling the right
    people?** chart and rate cards (`calling_right_people`) show that way's 100 calls: who they'd
    reach and who they'd miss. The outcomes come from `rule_game.three_ways_outcomes()`; at random
    has no list of its own, so it shows the counts expected on average. The box is a fragment
    (`compare_ways`), so a click reruns just that part.
  - **Why does the model give a patient their risk?**: a SHAP waterfall for one upcoming
    appointment, always on show, with big buttons to swap between a high-risk (70% or more, the
    high-risk flag), medium-risk (20% to 70%, forecast as a DNA but not flagged) and low-risk
    (under 5%) patient (`RISK_GROUPS`), and **Show a different patient** for another from the
    same group. The page opens on the same high-risk patient for every visitor (a fixed seed).
    The "How to read this chart" guide sits in a column beside it. A fragment
    (`explain_a_patient`).
  - A button leads on to the phone tab.
- **How many people should we phone?** tab: choose in the sidebar who an admin team would phone, by the number of calls
  the team can make each week or by minimum predicted risk. A 1,000-marker chart
  (ported from lambda_ml_streamlit_exercise) shows, for past appointments, how many calls would
  reach someone who then did not attend, and how many such patients would get no call. A panel
  near the top shows what happened in the real Physiotherapy pilot (see **The real pilot**,
  below). An effect slider, starting at the pilot's 80%, shows how many missed appointments the
  calls might avoid; the tab says the pilot had no comparison group, so other settings may see
  less. A "What this setting costs and might save" section turns the weekly figures into staff
  hours, the cost of the calls, the value of the missed appointments they reach and might avoid,
  and the **break-even effect**: the share of people reached a call would need to get to attend
  to cover its own staff time. It also compares appointments saved per 3 calls with the pilot's
  2, and estimates the early cancellations (slots freed for other patients) at the pilot's rate.
  Phoning everyone at 70% or more (the high-risk flag) with the 80% effect gives about 2 saved
  per 3 calls, as in the pilot.
- **Bonus: beat the model** tab: a short game to show how a machine learning model learns to
  beat a person's rule.
  - Built for a 2-5 minute visit to a stand: the setup is three numbers and a question, so the
    clue buttons are on screen without scrolling at 1920x1080 (100% and 125% scaling).
  - Using 1,000 past appointments (79 missed) and 100 calls, the visitor builds a rule for who
    to phone from the twelve clues the model sees (including *riskiest times of day*, *longer
    journeys*, *a longer gap since last seen* and *treatment already finished*), with each
    chosen clue getting an equal say.
    After each change, a callout says whether the last clue added or taken away made the list
    better or worse, and the rule is ranked against all 4,095 rules the buttons can make (the
    best combine six or more clues and reach 34 of 79; *most missed appointments before* on its
    own reaches 23).
  - They then watch a model learn its own weighting from 0 to 22,500 past appointments: the
    race against their best rule beside how much say each clue gets. A toast and an
    "Overtook you" marker show where the model passes them (800 for the best rule).
  - The verdict follows at once: a banner saying where the model overtook them, then two short
    cards: same clues but the model learnt which matter most (no rule from the buttons can match
    its 42), and nobody, not even the best computer model, gets them all (a risk is like a
    chance of rain: some high-risk patients turn up and some low-risk ones miss). The race chart
    shows only the model's line and the visitor's best rule: the best possible list (knowing
    each patient's true chance, 55 of 79, still in `rule_game_learning.json` as `perfect`) was
    taken off as too abstract for a stand. A short "Why a computer?" section comes last.
  - **Start again** (at the top once a game has begun, and after the verdict) clears the game
    for the next visitor.
  - The verdict names the best rule the buttons can make and how far the model is ahead of it.
    It says a simple rule does most of the work only when the model is 3 or fewer ahead, so it
    stays true if the data changes.
  - Two buttons lead on to the demo dashboard and back to How many people should we phone?, as in
    the geographic mini. The tabs are keyed (`key="page"`) so a button can switch them.
- **Try a demo dashboard** tab (last): the real tool's screens on synthetic data, with three views
  picked by the stacked switch at the top of the sidebar: **All appointments**, **Weekly summary**
  and **Next week's call list**. Last on the bar but drawn before the phone tab and the game in
  the script, as the game's training animation must come last.
  All appointments puts the table, the thing to click, as high on the page as it can:
  - A short introduction and the three forecast cards, beside the demonstration notice:
    forecast DNAs, confident DNAs (the best return on staff time when only a few calls can be
    made) and the value of appointments kept if every forecast DNA is phoned
    (`forecast_savings`; the value of clinic slots, not cash).
  - The appointment table, highest risk first, 10 rows at a time (`TABLE_ROWS`); it scrolls.
    Click a row to open an overlay (`explain_appointment`, an `st.dialog`) with a SHAP waterfall
    explaining that appointment's predicted risk of not attending and a "How to read this chart"
    guide (a small made-up example). A switch at the top of the overlay changes it to **Overall**:
    what the model relies on most (mean |SHAP|) across all upcoming appointments, with a second
    chart beside it for the filtered appointments when the sidebar filters leave at least 30
    (`overall_importance`). The switch is a segmented control, not `st.tabs`, because style.css
    turns every top-level tab bar into the page navbar. The overlay opens only on the click, not
    on later reruns, and dismissing it clears the selection so the same row can be clicked again.
    The table is a fragment (`appointments_table`), so a click reruns just that part and the
    overlay opens in about a second; the sidebar filters are outside it and rerun the page. Its
    DNA Risk and Confidence filters sit above it, on one row with a "What does confidence mean?"
    popover, and narrow only the table: in the sidebar they would fight the forecast cards and
    call list, and the chart below would go round in circles (they select by the forecast it
    explains).
  - **Weekly summary**: appointments and forecast DNAs for every upcoming week, by weekday,
    coloured by either list. Click a day to list its appointments under the grid, highest risk
    first. The call list view covers just next week.
  - **Next week's call list**: what the admin team would work from, earliest first, with each
    patient's name and appointment time.

The sidebar filters apply to the demo dashboard only (there is no date filter: click a day in the
Weekly summary view instead). The other tabs use all the upcoming appointments and hide them: How
many people should we phone? shows its own controls (who to phone, and the assumed effect of a call)
and the cost assumptions, and Why use a model? only the cost assumptions (they set the £ in its
tiles). The filters keep their settings while another tab is open (the script checks the open tab,
which it can since the tabs are keyed). The bonus game uses past appointments only, so the sidebar
is hidden while that tab (the third) is open (a CSS rule in `style.css` keyed on the selected tab,
so the sidebar's settings are kept for the other tabs).

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
- **Missed appointments:** a cost by specialty and New / Follow Up, from £100 to £260. These are
  illustrative values, loosely in line with NHS Payment Scheme outpatient prices: replace them
  with your Trust's own costs. Physiotherapy is £115, the real pilot's £21,000 over 182
  appointments.
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

The appointments belong to a fictional trust, Riverbourne Hospitals (`TRUST_NAME` in
`demo_data.py`), with four made-up sites. These replaced real hospital names one for one: each
site kept its place in `SITES` and `LOCATIONS`, its number of locations and its alphabetical
order, so the random draws, the model and every number stayed exactly the same.

They are made offline by `build/train_and_explain.py`, which follows the pipeline in
lambda_ml_streamlit_exercise:

1. `demo_data.generate_appointments()` makes up past appointments with a hidden "true" chance of not attending,
   and draws whether each patient attended. The chances average `DNA_RATE` (8%: recent NHS
   outpatient figures are about 6.4%, so the demo Trust is a little worse than average). The
   strongest clues are the real model's common reasons (see **The real pilot**): long journeys,
   long gaps since last seen and appointments after treatment had finished (below), alongside
   missed appointments before (`PREVIOUS_DNA_EFFECT`; a small group of patients miss again and
   again, `HABITUAL_SHARE`, `HABITUAL_MEAN`), against a lot that no clue can see (`NOISE_SD`).
   These are set so the model matches the real pilot: a test AUC near 0.86, and about 84% of the
   appointments it flags as high risk missed without a call. They were found by searching on the
   build's fixed seeds, and are sensitive: only about 60 past appointments are flagged, so a
   small change moves the 84% by up to 10 points and the AUC by about 0.01. Rerun the build
   after any change.
2. An XGBoost model is trained on those past appointments. Its trees are two deep, not three as
   in lambda_ml_streamlit_exercise: the demo's risks mostly add up clue by clue, and deeper trees
   fitted them less well than the bonus game's simpler model.
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
name is shown only on the call list (Next week's call list), with the appointment time. The model sees
the hour (`Time_Slot`, 08:00 to 16:00) as a clue: appointments starting 10-12 or 15-17 miss more
(`TIME_EFFECT` in `demo_data.py`), and Monday before midday gets an extra push
(`MONDAY_MORNING_EFFECT`) on top of Monday's own, so it is the worst slot of the week (about 15%
missed, against 7% for the rest of the week). The build checks this. Explanations show the real time
(`Appointment_Time`), not the hour slot.

**Predicted journey time.** Long or difficult journeys were one of the real model's common reasons
for a prediction, so each appointment has a predicted journey time (`Journey_Time_Mins`). Each patient
has a usual journey to their usual site (about 25 minutes typically, a little longer from more
deprived areas), and an appointment at another site takes 20 to 40 minutes more. Longer journeys
push the risk up (`JOURNEY_EFFECT`), more slowly the longer they get. It has its own random stream,
so adding it changed only the risk. The build checks that journeys of 45 minutes or more miss more
often (about 15%, against 6%) and that journey time is one of the model's four main reasons (it is
second, after missed appointments before). It shows in the explanations and as a clue in Your rule
vs the model, not as a column in the appointment tables.

**Gap since last seen, and treatment already finished.** The real model's other common reasons
were long waits between appointments and appointments after treatment had finished. Each
appointment has the weeks since the patient was last seen in that specialty, or, for a new
appointment, since they were referred (`Weeks_Since_Last`, "Weeks since last seen (or
referred)"); longer gaps push the risk up (`GAP_EFFECT`). About 3% of follow-ups come after the
course of treatment had finished (`Treatment_Ended`, `TREATMENT_ENDED_SHARE`), a strong push
(`TREATMENT_ENDED_EFFECT`). A small group of patients have "moved on" (`MOVED_ON_SHARE`): their
follow-ups are much more often after treatment had finished and their gaps longer, so these
signals pile up in the same patients and give some of the very high risks a call list needs.
Both have their own random stream, so adding them changed only the risk. The build checks that
the longest fifth of gaps miss more (about 16%, against 6%), that appointments after treatment
had finished miss much more (about 39%, against 6%), that both are among the model's five main
reasons (third and fourth), and that no one reason outweighs the others (missed appointments
before is now about 1.4 times the next, journey time; it was over three times before these were
added). Like journey time, they show in the explanations and as clues in the bonus game, not as
columns in the appointment tables.

Dates are not stored. `demo_data.load_forecast()` dates the appointments from next Monday onwards,
so they always roll forward from today while each appointment keeps its weekday.

To rebuild (for example after changing `SEED` in `build/train_and_explain.py` or the effects in
`demo_data.py`):

```
pip install -r build/requirements-build.txt
python build/train_and_explain.py
python build/rule_game_data.py
```

`build/rule_game_data.py` makes the two files for the bonus game tab (and the first tab's "Why not just phone those who've missed before?" box). It uses the same
split as the main build, and checks that its test appointments match `past_predictions.csv`.

- `rule_game_past.csv`: the past test appointments with a "phone first" score per clue, as a
  sensible person would use it. Numbers point the way that goes with more missed appointments,
  and categories are scored by last year's miss rate.
- `rule_game_learning.json`: the model's results after learning from 0, 25, 50, ... 22,500 past
  appointments.

The model on this tab is a logistic regression on raw inputs, so it has to learn which way each clue
points as well as how much it counts. That's why it starts out worse than a person. With all the
data it reaches the same share of misses as the XGBoost model, within 3 points (the build checks
this; 54% against 52%). The demo data has only one interaction between clues, Monday mornings
(below), which the simpler model can't learn, but it is a small part of the risk, so the simpler
model loses almost nothing.

The build checks that the SHAP values add up to each prediction, that the past DNA rate is
within a point of `DNA_RATE` and that every risk band has some appointments. It also checks the
match with the real pilot: a test AUC from 0.85 to 0.88 (near the "around 86%" the real model is
quoted at; it is 0.875), 80% to 88% of high-risk past appointments missed (the pilot's 84%), and
at least 1.2% of upcoming appointments flagged. It prints how the
forecasts fall into the risk bands and the high-risk flag.

It also checks **calibration** on the past appointments the model didn't train on: in each risk
band, the average predicted risk must be close to the share who really missed. The app adds risks
up (expected missed appointments, costs), so they must be true probabilities. That's why the
demo model isn't class-weighted. The live model trains with `scale_pos_weight=4`, which makes its
risks roughly four times the true odds, and that is why its yes/no line (50%) and bands (15/50/80%)
sit higher than the demo's.

## Risk bands and the high-risk flag

- **Risk bands** (`DNA_Risk()` in `streamlit_app.py`): Low under 5%, Medium 5% to 20%, High 20% to
  70%, Very High 70% or more.
- **Yes/no forecast** ("Expected DNA", `FORECAST_MIN` in the build and the app): a predicted risk
  of 20% or more, about two and a half times the average. The live model says yes at 50%, which
  with its `scale_pos_weight=4` is about a 20% true risk. That's about 10% of appointments; about
  4 in 10 of those past patients missed, and they include about half of everyone who missed.
- **High-risk flag** (`HIGH_RISK_MIN`): the Very High band, 70% or more, the live app's flag. About
  1.4% of upcoming and past appointments; about 84% of those past patients missed (the real
  pilot's high-risk patients: 84%), and they include about 15% of everyone who missed.
- **Weekly summary colours** (`weekly.MEASURES`): green up to 1% and red from 4% for the flag,
  and green up to 7% and red from 20% for the yes/no forecast. Set so that, over all the upcoming
  days, about 7% are red for either list and most of the rest amber or green, as before the
  data's factors were rebalanced (it was 2% and 6%, and 5% and 15%).

## The real pilot

The figures in `pilot.py` are from the real tool's pilot in a Physiotherapy service, from 24
December 2025 to 2 February 2026. Staff phoned 273 patients the model flagged as high risk, to
confirm attendance, rearrange if needed and reassure.

- These patients typically missed 84% of their appointments. After a call, 17% missed; among those
  who answered and confirmed they'd come, 8%, about the same as patients at low risk.
- 182 more appointments were attended, about £21,000 of revenue not lost: two appointments saved
  for every three calls. The calls also brought 29 early cancellations, slots offered to others.
- So of about 229 patients who would have missed, 182 attended: an effect of about 80%, the phone
  tab's starting value (`DEFAULT_CALL_EFFECT`, worked out from `pilot.EFFECT`).

The demo is set to match: the model's test AUC is about 0.87 (the real model is quoted as
"accurately predicting in around 86% of cases"), the high-risk flag catches patients who miss about
84% of the time, Physiotherapy is one of the specialties, and the real model's common reasons
(long or difficult journeys, long waits between appointments, appointments after treatment had
finished) are among the model's main reasons. The pilot had no comparison group
(it compared these patients with how often they usually missed), so the tab says the true effect
may be lower, especially lower down the list or in other services.

To go back to live data, replace `load_forecast()` in `demo_data.py` with a read of the combined
forecast view. You would also need SHAP values from the live model for the waterfalls.
