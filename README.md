# Outpatient DNA Forecast - demo build

Runs entirely on synthetic data. No SQL Server or .env file needed.

Based on https://github.com/WX-BIU/Outpatient-DNAs.

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What's in the app

- **Appointments** tab: a short introduction to the app, then the filterable appointment list.
  Click a row to see a SHAP waterfall explaining that appointment's predicted risk of not
  attending, with a "How to read this chart" guide. With no row selected, a bar chart shows
  what the model relies on most (mean |SHAP|) across all upcoming appointments, with a second
  chart beside it for the filtered appointments when the appointment-detail filters leave at
  least 30 (not shown for the DNA Risk and Confidence filters, which select by the forecast).
- **Weekly summary** tab: appointments and forecast DNAs by week and day.
- **Who would we phone?** tab: choose who an admin team would phone, by the number of calls
  the team can make each week or by minimum predicted risk. A 1,000-marker chart
  (ported from lambda_ml_streamlit_exercise) shows, for past appointments, how many calls would
  reach someone who then did not attend, and how many such patients would get no call. An
  assumed-effect slider shows how many missed appointments the calls might avoid. We don't know
  that a call changes whether someone attends, and the tab says so. A "What might the calls cost,
  and save?" section turns the weekly figures into staff hours, the cost of the calls, the value
  of the missed appointments they reach and might avoid, and the **break-even effect**: the share
  of people reached a call would need to get to attend to cover its own staff time.

The Appointments and Weekly summary tabs also show the expected cost of missed appointments
(each predicted risk times that appointment's cost).

The sidebar filters apply to the first two tabs. On the third they change only the upcoming
calls-per-week figures, since the chart uses past appointments.

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

The app reads four precomputed files in `data/`:

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
   probability minus the second, which for a yes/no model is |2 × risk − 1|.
4. `shap.TreeExplainer` explains each prediction on the probability scale.

Dates are not stored. `demo_data.load_forecast()` dates the appointments from next Monday onwards,
so they always roll forward from today while each appointment keeps its weekday.

To rebuild (for example after changing `SEED` in `build/train_and_explain.py` or the effects in
`demo_data.py`):

```
pip install -r build/requirements-build.txt
python build/train_and_explain.py
```

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
