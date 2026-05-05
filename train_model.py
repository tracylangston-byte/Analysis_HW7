"""
train_model.py
--------------
Fits the chosen model on the training period and optionally validates on the
test period. Run via run_workflow.sh, or directly:

    python train_model.py --email YOU@EMAIL.COM --pin 1234 [--other-options]
"""

import os
import argparse
import pandas as pd
import hf_hydrodata
from forecast_functions import (
    get_training_test_data,
    fit_longterm_avg_model,
    fit_monthly_avg_model,
    make_monthly_avg_predictions,
    compute_metrics,
    plot_validation,
    save_model,
    load_model,
    get_met_data_single_cell,
    fit_met_regression_model,
    validate_met_regression_model,
)

parser = argparse.ArgumentParser()
parser.add_argument('--email',       required=True)
parser.add_argument('--pin',         required=True)
parser.add_argument('--gauge-id',    default='09506000')
parser.add_argument('--ar-order',    type=int, default=7)
parser.add_argument('--train-start', default='1990-01-01')
parser.add_argument('--train-end',   default='2022-12-31')
parser.add_argument('--test-start',  default='2023-01-01')
parser.add_argument('--test-end',    default='2024-12-31')
parser.add_argument(
    '--model',
    default='longterm_avg',
    choices=['longterm_avg', 'monthly_avg', 'met_regression']
)
parser.add_argument('--refit',       default='True')
parser.add_argument('--validate',    default='True')
args = parser.parse_args()

REFIT_MODEL    = args.refit.lower()    == 'true'
RUN_VALIDATION = args.validate.lower() == 'true'

hf_hydrodata.register_api_pin(email=args.email, pin=args.pin)

print("\n--- Step 1: Download streamflow data ---")
train, test = get_training_test_data(
    args.gauge_id, args.train_start, args.train_end,
    args.test_start, args.test_end
)

# ── Long-term average model ───────────────────────────────────────────────────
if args.model == 'longterm_avg':
    print("\n--- Step 2: Fit long-term average model ---")
    if REFIT_MODEL or not os.path.exists('saved_model.pkl'):
        mean_flow = fit_longterm_avg_model(train)
        print(f"  Long-term mean: {mean_flow:.2f} cfs")
        save_model(mean_flow)
    else:
        mean_flow = load_model()
        if not isinstance(mean_flow, float):
            raise TypeError(
                "saved_model.pkl does not contain a longterm_avg model. "
                "Re-run with --refit True --model longterm_avg to train one."
            )

    if RUN_VALIDATION:
        print("\n--- Step 3: Validate on test period ---")
        train_fitted    = pd.Series(mean_flow, index=train.index)
        forecast_series = pd.Series(mean_flow, index=test.index)

        metrics = compute_metrics(test['streamflow_cfs'].values, forecast_series.values)
        print("\n  Validation metrics:")
        for name, val in metrics.items():
            print(f"    {name:<12}: {val:.4f}")
        print("\n  NSE guide: >0.75 very good | 0.65–0.75 good | "
              "0.50–0.65 satisfactory | <0.50 poor")

        print("\n  Generating validation plot ...")
        plot_validation(
            train['streamflow_cfs'], test['streamflow_cfs'],
            forecast_series, metrics, 'Long-term Average',
            train_forecast_cfs=train_fitted
        )

# ── Monthly average model ─────────────────────────────────────────────────────
if args.model == 'monthly_avg':
    print("\n--- Step 2: Fit monthly average model ---")

    if REFIT_MODEL or not os.path.exists('saved_model.pkl'):
        monthly_model = fit_monthly_avg_model(train)
        print("  Monthly mean streamflows:")
        for month, flow in monthly_model.items():
            print(f"    Month {month:02d}: {flow:.2f} cfs")
        save_model(monthly_model)
    else:
        monthly_model = load_model()
        if not isinstance(monthly_model, dict):
            raise TypeError(
                "saved_model.pkl does not contain a monthly_avg model. "
                "Re-run with --refit True --model monthly_avg to train one."
            )

    if RUN_VALIDATION:
        print("\n--- Step 3: Validate on test period ---")

        train_fitted = make_monthly_avg_predictions(monthly_model, train.index)
        forecast_series = make_monthly_avg_predictions(monthly_model, test.index)

        metrics = compute_metrics(test['streamflow_cfs'].values, forecast_series.values)

        print("\n  Validation metrics:")
        for name, val in metrics.items():
            print(f"    {name:<12}: {val:.4f}")

        print("\n  NSE guide: >0.75 very good | 0.65–0.75 good | "
              "0.50–0.65 satisfactory | <0.50 poor")

        print("\n  Generating validation plot ...")
        plot_validation(
            train['streamflow_cfs'],
            test['streamflow_cfs'],
            forecast_series,
            metrics,
            'Monthly Average',
            train_forecast_cfs=train_fitted
        )

# ── Meteorological regression model ───────────────────────────────────────────
if args.model == 'met_regression':
    print("\n--- Step 2: Download precipitation and temperature data ---")

    # Start earlier so lagged 7-day predictors are available.
    met_start = (
        pd.Timestamp(args.train_start) - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")

    met = get_met_data_single_cell(
        date_start=met_start,
        date_end=args.test_end
    )

    print(f"  Met data: {met.index[0].date()} to {met.index[-1].date()}")

    print("\n--- Step 3: Fit met_regression model ---")

    if REFIT_MODEL or not os.path.exists('saved_model.pkl'):
        met_model = fit_met_regression_model(train, met)
        save_model(met_model)
    else:
        met_model = load_model()
        if not isinstance(met_model, dict) or met_model.get("model_type") != "met_regression":
            raise TypeError(
                "saved_model.pkl does not contain a met_regression model. "
                "Re-run with --refit True --model met_regression to train one."
            )

    if RUN_VALIDATION:
        print("\n--- Step 4: Validate on test period ---")

        train_fitted, forecast_series, test_obs = validate_met_regression_model(
            met_model,
            train,
            test,
            met
        )

        metrics = compute_metrics(test_obs.values, forecast_series.values)

        print("\n  Validation metrics:")
        for name, val in metrics.items():
            print(f"    {name:<12}: {val:.4f}")

        print("\n  NSE guide: >0.75 very good | 0.65–0.75 good | "
              "0.50–0.65 satisfactory | <0.50 poor")

        print("\n  Generating validation plot ...")
        plot_validation(
            train['streamflow_cfs'],
            test_obs,
            forecast_series,
            metrics,
            'Met Regression',
            train_forecast_cfs=train_fitted
        )

print("\nTraining complete.")
