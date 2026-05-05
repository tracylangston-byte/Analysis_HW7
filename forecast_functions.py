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

def fit_monthly_avg_model(train_df):
    """Return a dictionary mapping each calendar month (1-12) to the mean streamflow
    for that month over the training period."""
    return train_df.groupby(train_df.index.month)['streamflow_cfs'].mean().to_dict()

def make_5day_forecast_longterm(mean_flow, forecast_date, n_days=5):
    """Return DataFrame with the long-term mean flow for every forecast day."""
    dates = pd.date_range(start=forecast_date, periods=n_days, freq='D')
    return pd.DataFrame({'Forecast_cfs': mean_flow}, index=dates)

def make_5day_forecast_monthly(monthly_means, forecast_date, n_days=5):
    """Return a DataFrame with Forecast_cfs indexed by date, where each day's forecast
    is the historical mean for that day's calendar month. To look up the month for
    a date d, use d.month."""
    return pd.DataFrame({'Forecast_cfs': mean_flow}, index=dates)


# ── Meteorological regression model ───────────────────────────────────────────

def get_met_data_single_cell(date_start, date_end,
                             lat=34.4483605, lon=-111.7898705,
                             dataset="CW3E", grid="conus2"):
    """
    Download precipitation and air temperature from one gridded forcing cell
    near the Verde River near Camp Verde gauge.

    Precipitation is returned as daily total precipitation.
    Air temperature is returned as daily average air temperature.

    Notes:
    - CW3E forcing is hourly.
    - precipitation is treated as a rate, so daily precipitation is calculated
      by summing hourly values * 3600 seconds/hour.
    - air_temp is usually in Kelvin, but for regression the units are okay
      as long as they are consistent.
    """
    start_ts = pd.Timestamp(date_start)
    end_ts = pd.Timestamp(date_end)

    # Convert the gauge latitude/longitude to HydroFrame grid indices.
    grid_i, grid_j = hf_hydrodata.to_ij(grid, lat, lon)
    grid_bounds = (grid_i, grid_j, grid_i + 1, grid_j + 1)

    forcing = {}

    for var in ["precipitation", "air_temp"]:
        options = {
            "dataset": dataset,
            "dataset_version": "1.0",
            "variable": var,
            "temporal_resolution": "hourly",
            "grid": grid,
            "start_time": start_ts.strftime("%Y-%m-%d"),
            # Add one day because HydroData end_time is effectively the end boundary
            "end_time": (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "grid_bounds": grid_bounds,
        }

        arr = hf_hydrodata.get_gridded_data(options).squeeze()
        forcing[var] = np.asarray(arr).astype(float)

    # Build an hourly time index matching the returned data length.
    n_hours = min(len(forcing["precipitation"]), len(forcing["air_temp"]))
    hourly_index = pd.date_range(start=start_ts, periods=n_hours, freq="h")

    hourly = pd.DataFrame(
        {
            "precip_rate": forcing["precipitation"][:n_hours],
            "air_temp": forcing["air_temp"][:n_hours],
        },
        index=hourly_index
    )

    # Convert hourly precipitation rate to a daily total.
    # If the precipitation units are mm/s, this gives mm/hr before daily summing.
    daily = pd.DataFrame()
    daily["precip_1day"] = (hourly["precip_rate"] * 3600).resample("D").sum()
    daily["air_temp_1day"] = hourly["air_temp"].resample("D").mean()

    return daily


def make_met_regression_features(flow_df, met_df):
    """
    Combine streamflow and meteorological data into regression features.

    The target is today's log streamflow.
    The predictors use only information from previous days.
    """
    df = flow_df[["streamflow_cfs", "log_flow"]].copy()
    df = df.join(met_df, how="inner")

    # Lagged streamflow features
    df["flow_lag1"] = df["streamflow_cfs"].shift(1)
    df["flow_7day_mean"] = df["streamflow_cfs"].shift(1).rolling(7).mean()

    # Lagged meteorological features
    df["precip_7day_sum"] = df["precip_1day"].shift(1).rolling(7).sum()
    df["temp_7day_mean"] = df["air_temp_1day"].shift(1).rolling(7).mean()

    # Seasonal timing features
    day_of_year = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    df = df.dropna()

    feature_cols = [
        "flow_lag1",
        "flow_7day_mean",
        "precip_7day_sum",
        "temp_7day_mean",
        "sin_doy",
        "cos_doy",
    ]

    X = df[feature_cols]
    y = df["log_flow"]

    return X, y, df


def fit_met_regression_model(train_df, met_df):
    """
    Fit a simple linear regression model using numpy.

    This avoids adding another package dependency. The model predicts log_flow,
    then forecast values are converted back to cfs.
    """
    X, y, _ = make_met_regression_features(train_df, met_df)

    feature_cols = list(X.columns)

    # Standardize features so variables with large units do not dominate.
    x_mean = X.mean()
    x_std = X.std().replace(0, 1)

    X_scaled = (X - x_mean) / x_std

    # Add intercept column
    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled.values])

    # Least-squares regression
    coef, _, _, _ = np.linalg.lstsq(X_design, y.values, rcond=None)

    model = {
        "model_type": "met_regression",
        "feature_cols": feature_cols,
        "x_mean": x_mean.to_dict(),
        "x_std": x_std.to_dict(),
        "coef": coef,
    }

    return model


def predict_met_regression(model, X):
    """
    Predict streamflow in cfs from met_regression model features.
    """
    feature_cols = model["feature_cols"]

    x_mean = pd.Series(model["x_mean"])
    x_std = pd.Series(model["x_std"])

    X_use = X[feature_cols]
    X_scaled = (X_use - x_mean) / x_std

    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled.values])
    pred_log = X_design @ model["coef"]

    # Convert log(flow + 1) back to flow
    pred_cfs = np.exp(pred_log) - 1

    # Avoid negative predictions caused by regression math
    pred_cfs = np.maximum(pred_cfs, 0)

    return pd.Series(pred_cfs, index=X.index)


def validate_met_regression_model(model, train_df, test_df, met_df):
    """
    Create fitted values for train and predictions for test using observed
    previous-day data. This is validation, not future forecasting.
    """
    combined = pd.concat([train_df, test_df]).sort_index()

    X_all, _, _ = make_met_regression_features(combined, met_df)
    pred_all = predict_met_regression(model, X_all)

    train_pred = pred_all.loc[pred_all.index.intersection(train_df.index)]
    test_pred = pred_all.loc[pred_all.index.intersection(test_df.index)]

    test_obs = test_df.loc[test_pred.index, "streamflow_cfs"]

    return train_pred, test_pred, test_obs


def make_5day_forecast_met_regression(model, recent_flow_df, recent_met_df,
                                      forecast_date, n_days=5):
    """
    Make a 5-day forecast using only data before the forecast date.

    Since we are not using forecasted precipitation or temperature, the model
    carries forward the most recent 7-day meteorological summary for all 5
    forecast days. Streamflow is updated recursively using the previous
    predicted flow.
    """
    forecast_dates = pd.date_range(start=forecast_date, periods=n_days, freq="D")

    history = recent_flow_df[["streamflow_cfs"]].copy()
    met_history = recent_met_df.copy()

    forecasts = []

    for date in forecast_dates:
        # Use streamflow history available up to the day before this forecast date.
        flow_lag1 = history["streamflow_cfs"].iloc[-1]
        flow_7day_mean = history["streamflow_cfs"].iloc[-7:].mean()

        # Use met data only from before the forecast date.
        precip_7day_sum = met_history["precip_1day"].iloc[-7:].sum()
        temp_7day_mean = met_history["air_temp_1day"].iloc[-7:].mean()

        sin_doy = np.sin(2 * np.pi * date.dayofyear / 365.25)
        cos_doy = np.cos(2 * np.pi * date.dayofyear / 365.25)

        X = pd.DataFrame(
            {
                "flow_lag1": [flow_lag1],
                "flow_7day_mean": [flow_7day_mean],
                "precip_7day_sum": [precip_7day_sum],
                "temp_7day_mean": [temp_7day_mean],
                "sin_doy": [sin_doy],
                "cos_doy": [cos_doy],
            },
            index=[date]
        )

        forecast_cfs = float(predict_met_regression(model, X).iloc[0])
        forecasts.append(forecast_cfs)

        # Add predicted flow to history so the next forecast day can use it.
        history.loc[date, "streamflow_cfs"] = forecast_cfs

        # Do NOT add future met data. This keeps the model from using future
        # precipitation or temperature forecasts.

    return pd.DataFrame({"Forecast_cfs": forecasts}, index=forecast_dates)

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
    # plt.show()


def save_model(model, path='saved_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to {path}")


def load_model(path='saved_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Model loaded from {path}")
    return model

def fit_monthly_avg_model(train_df):
    """
    Fit a monthly-average model.

    This groups the training data by month and calculates the mean streamflow
    for each month of the year. The result is a dictionary like:
    {1: January mean flow, 2: February mean flow, ...}
    """
    monthly_means = train_df.groupby(train_df.index.month)['streamflow_cfs'].mean().to_dict()
    return monthly_means


def make_5day_forecast_monthly(monthly_model, forecast_date, n_days=5):
    """
    Make a 5-day forecast using the average streamflow for each forecast month.
    """
    dates = pd.date_range(start=forecast_date, periods=n_days, freq='D')
    forecasts = []

    for date in dates:
        month = date.month
        forecast = monthly_model[month]
        forecasts.append(forecast)

    return pd.DataFrame({'Forecast_cfs': forecasts}, index=dates)


def make_monthly_avg_predictions(monthly_model, dates):
    """
    Make monthly-average predictions for an existing set of dates.
    This is useful for validation on the test period.
    """
    predictions = []

    for date in dates:
        month = date.month
        prediction = monthly_model[month]
        predictions.append(prediction)

    return pd.Series(predictions, index=dates)
