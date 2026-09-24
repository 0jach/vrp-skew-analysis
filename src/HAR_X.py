"""Expected realized volatility estimated with a HAR-X model."""

import numpy as np
import pandas as pd


def compute_har_x_rv(
    returns: pd.DataFrame,
    skew: pd.Series | pd.DataFrame,
    min_training: int = 252,
) -> pd.Series:
    """Return point-in-time 22-observation HAR-X volatility forecasts.

    The model predicts future annualized variance from daily, weekly, and
    monthly realized variance plus contemporaneous downside skew.
    """
    daily_returns = returns.set_index("date")["return"].sort_index()
    daily_variance = 252 * daily_returns.pow(2)

    if isinstance(skew, pd.DataFrame):
        skew = skew.set_index("date")["skew"]
    skew = skew.sort_index().reindex(daily_variance.index)

    features = pd.DataFrame(
        {
            "daily": daily_variance,
            "weekly": daily_variance.rolling(5).mean(),
            "monthly": daily_variance.rolling(22).mean(),
            "skew": skew,
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
        # The target originating 22 observations earlier is the latest one
        # whose future variance is fully known at this date.
        training_end = position - 22
        if (
            training_end < 0
            or cumulative_n[training_end] < min_training
            or not np.isfinite(x[position]).all()
        ):
            continue

        coefficients = np.linalg.lstsq(cumulative_xx[training_end], cumulative_xy[training_end], rcond=None)[0]
        prediction = x[position] @ coefficients
        forecast.iloc[position] = np.sqrt(max(prediction, 0.0))

    return forecast
