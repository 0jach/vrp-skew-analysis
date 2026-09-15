# %%
import numpy as np
import pandas as pd

from data_parser import extract_ret


def build_weekly_future_rv(daily_returns):
    daily = daily_returns.reset_index()[["date", "return"]].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)


    trading_dates = daily["date"].to_numpy(dtype="datetime64[ns]")

    # Target each Wednesday and move forward to the next trading day when closed.
    wednesdays = pd.date_range(
        daily["date"].min(), daily["date"].max(), freq="W-WED"
    ).to_numpy(dtype="datetime64[ns]")
    observation_idx = np.searchsorted(trading_dates, wednesdays, side="left")
    observation_idx = observation_idx[observation_idx < len(trading_dates)]
    observation_dates = trading_dates[observation_idx]

    # A label is valid only when its full 30-calendar-day horizon is observable.
    window_ends = observation_dates + np.timedelta64(30, "D")
    complete = window_ends <= trading_dates[-1]
    observation_dates = observation_dates[complete]
    window_ends = window_ends[complete]

    # searchsorted makes the label window exactly: t < date <= t + 30 days.
    window_starts = np.searchsorted(trading_dates, observation_dates, side="right")
    window_stops = np.searchsorted(trading_dates, window_ends, side="right")

    squared_return = daily["return"].fillna(0.0).to_numpy(dtype=float) ** 2
    cumulative_squared = np.r_[0.0, np.cumsum(squared_return)]

    sum_squared = cumulative_squared[window_stops] - cumulative_squared[window_starts]

    weekly = pd.DataFrame(
        {
            "date": pd.to_datetime(observation_dates),
            "future_RV_30d": np.sqrt(
                365 / 30 * sum_squared
            ),
        }
    )

    return weekly

# %%
returns = extract_ret()
weekly_rv = build_weekly_future_rv(returns)
print(weekly_rv)
