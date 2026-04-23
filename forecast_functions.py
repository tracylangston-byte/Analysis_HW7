"""
forecast_functions.py
---------------------
Helper functions shared by train_model.py and generate_forecast.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import hf_hydrodata


def get_training_test_data(gauge_id, train_start, train_end, test_start, test_end):
    """
    Download daily streamflow and split into train/test DataFrames.
    Both DataFrames have columns: streamflow_cfs, log_flow.
    """
    raw = hf_hydrodata.get_point_data(
        dataset="usgs_nwis",
        variable="streamflow",
        temporal_resolution="daily",
        aggregation="mean",
        site_ids=gauge_id,
        date_start=train_start,
        date_end=test_end
    )
    if 'date' in raw.columns:
        raw.index = pd.to_datetime(raw['date'])
    df = raw[[gauge_id]].rename(columns={gauge_id: 'streamflow_cfs'}).sort_index().dropna()
    df['log_flow'] = np.log(df['streamflow_cfs'] + 1)

    train = df.loc[train_start:train_end]
    test  = df.loc[test_start:test_end]
    print(f"  Training: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} days)")
    print(f"  Test:     {test.index[0].date()} to {test.index[-1].date()} ({len(test):,} days)")
    return train, test


def get_recent_data(gauge_id, forecast_date, ar_order):
    """
    Download recent observations before forecast_date and return as a DataFrame.
    Uses a metadata call to check the latest available date before downloading data.
    Raises ValueError if forecast_date is after the latest available data date,
    or if fewer than ar_order days of data precede the forecast date.
    """
    forecast_ts = pd.Timestamp(forecast_date)

    # Cheap metadata call to validate forecast_date without downloading the full record
    meta = hf_hydrodata.get_point_metadata(
        dataset="usgs_nwis",
        variable="streamflow",
        temporal_resolution="daily",
        aggregation="mean",
        site_ids=gauge_id
    )
    latest = pd.Timestamp(meta['last_date_data_available'].iloc[0])
    if forecast_ts > latest:
        raise ValueError(
            f"Forecast date {forecast_ts.date()} is after the latest available data "
            f"({latest.date()}). Choose a date on or before {latest.date()}."
        )

    # Download only the window needed: ar_order days for model seeding, 30 days for the plot
    n_days = max(ar_order, 30)
    date_start = (forecast_ts - pd.Timedelta(days=n_days)).strftime('%Y-%m-%d')
    date_end   = (forecast_ts - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    raw = hf_hydrodata.get_point_data(
        dataset="usgs_nwis",
        variable="streamflow",
        temporal_resolution="daily",
        aggregation="mean",
        site_ids=gauge_id,
        date_start=date_start,
        date_end=date_end
    )
    if 'date' in raw.columns:
        raw.index = pd.to_datetime(raw['date'])
    df = raw[[gauge_id]].rename(columns={gauge_id: 'streamflow_cfs'}).sort_index().dropna()
    df['log_flow'] = np.log(df['streamflow_cfs'] + 1)

    if len(df) < ar_order:
        raise ValueError(
            f"Need at least {ar_order} days before forecast date; only {len(df)} found."
        )
    return df


def fit_longterm_avg_model(train_df):
    """Return the mean streamflow (cfs) over the entire training period."""
    return float(train_df['streamflow_cfs'].mean())


def make_5day_forecast_longterm(mean_flow, forecast_date, n_days=5):
    """Return DataFrame with the long-term mean flow for every forecast day."""
    dates = pd.date_range(start=forecast_date, periods=n_days, freq='D')
    return pd.DataFrame({'Forecast_cfs': mean_flow}, index=dates)


def compute_metrics(observed_cfs, predicted_cfs):
    """Return dict with RMSE, R², and NSE (Nash-Sutcliffe Efficiency)."""
    obs  = np.array(observed_cfs)
    pred = np.array(predicted_cfs)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    r2   = np.corrcoef(obs, pred)[0, 1] ** 2
    nse  = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
    return {'RMSE (cfs)': rmse, 'R2': r2, 'NSE': nse}


def plot_validation(train_cfs, test_cfs, forecast_cfs, metrics, model_label,
                    train_forecast_cfs=None, save_path='validation_plot.png'):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    axes[0].plot(train_cfs.index, train_cfs.values,
                 color='steelblue', linewidth=0.6, alpha=0.7, label='Training')
    if train_forecast_cfs is not None:
        axes[0].plot(train_forecast_cfs.index, train_forecast_cfs.values,
                     color='tomato', linewidth=0.8, linestyle='--', alpha=0.8,
                     label=f'{model_label} Fitted (train)')
    axes[0].plot(test_cfs.index, test_cfs.values,
                 color='black', linewidth=1.0, label='Observed (test)')
    axes[0].plot(forecast_cfs.index, forecast_cfs.values,
                 color='tomato', linewidth=1.2, linestyle='--',
                 label=f'{model_label} Predicted (test)')
    axes[0].axvline(test_cfs.index[0], color='gray', linestyle=':', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Streamflow (cfs)')
    axes[0].set_title(f'{model_label} Validation — Verde River')
    axes[0].legend(fontsize=9)

    obs  = test_cfs.values
    pred = forecast_cfs.values
    # Restrict to finite, positive pairs — multi-step AR forecasts can diverge,
    # which would otherwise collapse all visible points to the origin on a linear scale.
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs_v, pred_v = obs[valid], pred[valid]
    if len(obs_v) > 0:
        lo = min(obs_v.min(), pred_v.min())
        hi = max(obs_v.max(), pred_v.max())
        axes[1].scatter(obs_v, pred_v, alpha=0.5, color='steelblue', s=15, edgecolors='none')
        axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='1:1 line')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
    axes[1].set_xlabel('Observed Streamflow (cfs)')
    axes[1].set_ylabel('Predicted Streamflow (cfs)')
    axes[1].set_title(
        f"Observed vs Predicted  |  "
        f"R² = {metrics['R2']:.3f},  NSE = {metrics['NSE']:.3f},  "
        f"RMSE = {metrics['RMSE (cfs)']:.1f} cfs"
    )
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {save_path}")
    plt.show()


def save_model(model, path='saved_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to {path}")


def load_model(path='saved_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Model loaded from {path}")
    return model
