# Outpatient DNA Forecast - demo build

Runs entirely on synthetic data. No SQL Server or .env file needed.

Based on https://github.com/WX-BIU/Outpatient-DNAs.

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What's in the app

- **Appointments** tab: the filterable appointment list. Click a row to see a SHAP waterfall
  explaining that appointment's predicted risk of not attending, with a "How to read this chart" guide.
- **Weekly summary** tab: appointments and forecast DNAs by week and day.

The sidebar filters apply to both tabs.

## Where the data and explanations come from

The app reads three precomputed files in `data/`:

- `forecast.csv`: the future appointments with the model's predicted DNA risk
- `shap_values.csv`: each appointment's SHAP contributions, in percentage points
- `model_meta.json`: the average (base) risk, test AUC and package versions

They are made offline by `build/train_and_explain.py`, which follows the pipeline in
lambda_ml_streamlit_exercise:

1. `demo_data.generate_appointments()` makes up past appointments with a hidden "true" chance of not attending,
   and draws whether each patient attended.
2. An XGBoost model is trained on those past appointments.
3. The model scores 3,200 future appointments.
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
