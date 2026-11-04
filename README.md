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
  that a call changes whether someone attends, and the tab says so.

The sidebar filters apply to the first two tabs. On the third they change only the upcoming
calls-per-week figures, since the chart uses past appointments.

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
   and draws whether each patient attended.
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

The build checks that the SHAP values add up to each prediction and that the share of forecast
DNAs and the spread of risk bands stay similar to the original demo.

To go back to live data, replace `load_forecast()` in `demo_data.py` with a read of the combined
forecast view. You would also need SHAP values from the live model for the waterfalls.
