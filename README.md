# Outpatient DNA Forecast - demo build

Runs entirely on synthetic data. No SQL Server, .env file or model files needed.

Based on https://github.com/WX-BIU/Outpatient-DNAs.

streamlit run

Change `seed` in `demo_data.generate()` for a different dummy dataset (dates always roll forward from today).
To go back to live data, replace `load_forecast()` in `app.py` with a read of the combined forecast view.
