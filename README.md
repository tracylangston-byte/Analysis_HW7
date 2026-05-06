# Analysis_HW7

Created for Analysis 2 Assignment 7.

# Verde River Streamflow Forecasting Workflow

This repository contains a Python workflow for generating 5-day daily average streamflow forecasts for the Verde River near Camp Verde, Arizona, using USGS gauge 09506000.

The workflow allows the user to choose a forecast date, select a model type, decide whether to refit the model, and decide whether to run model validation.

## Files in this repository

- `run_workflow.ps1` — Main workflow script. The user changes settings here and runs this file from the terminal.
- `train_model.py` — Trains the selected model and optionally runs validation.
- `generate_forecast.py` — Generates a 5-day streamflow forecast for the selected forecast date.
- `forecast_functions.py` — Contains helper functions used by the training and forecast scripts.
- `environment.yml` — Lists the Python environment needed to run the workflow.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate hw7
```

## Main user settings

The main workflow settings are located near the top of `run_workflow.ps1`.

- `GAUGE_ID` — USGS gauge ID. The default is `09506000`.
- `TRAIN_START` and `TRAIN_END` — Date range used to train the model.
- `TEST_START` and `TEST_END` — Date range held out for model validation.
- `FORECAST_DATE` — First day of the 5-day forecast.
- `REFIT_MODEL` — If `True`, the selected model is retrained.
- `RUN_VALIDATION` — If `True`, validation metrics and a validation plot are generated.
- `MODEL` — Model choices: `longterm_avg`, `monthly_avg`, or `met_regression`.

You may choose from three models: 
- `longterm_avg`: predicts the long-term average streamflow from the training period
- `monthly_avg`: predicts streamflow using the historical average for the forecast month
- `met_regression`: predicts streamflow using lagged streamflow, recent precipitation, recent air temperature, and seasonal timing

## Running the workflow

For Mac, Linux, or GitHub Codespaces, run:

```bash
bash run_workflow.sh
```

For Windows PowerShell, run:

```powershell
.\run_workflow.ps1
```

Both scripts use the same workflow settings: gauge ID, training dates, test dates, forecast date, model choice, refit option, and validation option.

The script will ask for your HydroFrame email and PIN, then train the selected model and/or generate a forecast depending on the settings.
