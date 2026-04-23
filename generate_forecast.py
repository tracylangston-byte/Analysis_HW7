"""
generate_forecast.py
--------------------
Generates a 5-day streamflow forecast starting on a user-specified date.
Run via run_workflow.sh, or directly:

    python generate_forecast.py --email YOU@EMAIL.COM --pin 1234 [--other-options]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import hf_hydrodata
from forecast_functions import (
    get_recent_data,
    make_5day_forecast_longterm,
    load_model,
)

parser = argparse.ArgumentParser()
parser.add_argument('--email',         required=True)
parser.add_argument('--pin',           required=True)
parser.add_argument('--gauge-id',      default='09506000')
parser.add_argument('--ar-order',      type=int, default=7)
parser.add_argument('--forecast-date', default='2024-04-30')
parser.add_argument('--model',         default='longterm_avg', choices=['longterm_avg'])
args = parser.parse_args()

forecast_date_ts = pd.Timestamp(args.forecast_date)

hf_hydrodata.register_api_pin(email=args.email, pin=args.pin)

print("\n--- Step 1: Download recent streamflow data ---")
recent = get_recent_data(args.gauge_id, args.forecast_date, args.ar_order)

# ── Long-term average model ────────────────────────────────────────────────────
if args.model == 'longterm_avg':
    print("\n--- Step 2: Load long-term average model ---")
    mean_flow = load_model()
    if not isinstance(mean_flow, float):
        raise TypeError(
            "saved_model.pkl does not contain a longterm_avg model. "
            "Re-run train_model.py with --refit True --model longterm_avg first."
        )
    print("\n--- Step 3: Generate 5-day long-term average forecast ---")
    forecast_df = make_5day_forecast_longterm(mean_flow, args.forecast_date)
    model_label = 'Long-term Average'

print(f"\n  5-Day Streamflow Forecast — Verde River ({model_label})")
print(f"  Starting: {forecast_date_ts.date()}\n")
print(f"  {'Date':<14}  Forecast (cfs)")
print(f"  {'-'*30}")
for date, row in forecast_df.iterrows():
    print(f"  {str(date.date()):<14}  {row['Forecast_cfs']:.1f}")

recent_cfs = recent['streamflow_cfs'].iloc[-30:]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(recent_cfs.index, recent_cfs.values,
        color='steelblue', linewidth=1.3, label='Recent Observed (30 days)')
ax.plot(forecast_df.index, forecast_df['Forecast_cfs'],
        'ro--', linewidth=1.5, markersize=6, label='5-Day Forecast')
ax.axvline(forecast_date_ts, color='gray', linestyle=':', linewidth=1.2)
ax.set_yscale('log')
ax.set_ylabel('Streamflow (cfs)')
ax.set_title(f'Verde River 5-Day Forecast  |  Starting {forecast_date_ts.date()}  ({model_label})')
ax.legend()
plt.tight_layout()
plt.savefig('forecast_plot.png', dpi=150, bbox_inches='tight')
print("  Plot saved to forecast_plot.png")
plt.show()
