"""Expected realized volatility estimated with a HAR model."""

import numpy as np
import pandas as pd

try:
    from .data_parser import load_clean_data
except ImportError:  # Allow running this file directly.
    from data_parser import load_clean_data


def compute_expct_rv(
    returns: pd.DataFrame | None = None, min_training: int = 252
) -> pd.Series:
    """Return point-in-time HAR forecasts of 22-day realized volatility."""
    if returns is None:
        _, returns = load_clean_data()

    daily_returns = returns.set_index("date")["return"].sort_index()
    daily_variance = 252 * daily_returns.pow(2)

    features = pd.DataFrame(
        {
            "daily": daily_variance,
            "weekly": daily_variance.rolling(5).mean(),
            "monthly": daily_variance.rolling(22).mean(),
        }
    )
    future_variance = daily_variance.rolling(22).mean().shift(-22)
    forecast = pd.Series(np.nan, index=features.index, name="expected_rv")

    x = np.column_stack([np.ones(len(features)), features.to_numpy()])
    y = future_variance.to_numpy()
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)

    xx = np.einsum("ni,nj->nij", x, x)
    xy = x * y[:, None]
    xx[~valid] = 0
    xy[~valid] = 0
    cumulative_xx = np.cumsum(xx, axis=0)
    cumulative_xy = np.cumsum(xy, axis=0)
    cumulative_n = np.cumsum(valid)

    for position in range(len(features)):
        # Row position - 22 is the latest target fully known at this date.
        training_end = position - 22
        if (
            training_end < 0
            or cumulative_n[training_end] < min_training
            or not np.isfinite(x[position]).all()
        ):
            continue

        coefficients = np.linalg.lstsq(
            cumulative_xx[training_end], cumulative_xy[training_end], rcond=None
        )[0]
        prediction = x[position] @ coefficients
        forecast.iloc[position] = np.sqrt(max(prediction, 0.0))

    return forecast


if __name__ == "__main__":
    print(compute_expct_rv().dropna().tail())
