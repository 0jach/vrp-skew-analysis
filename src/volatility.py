"""Realized volatility measures computed from observed returns."""

import numpy as np
import pandas as pd

try:
    from .data_parser import load_clean_data
except ImportError:  # Allow running this file directly.
    from data_parser import load_clean_data


ANNUALIZATION = 252
HORIZON_OBSERVATIONS = 22


def compute_actual_rv(
    returns: pd.DataFrame | None = None,
    horizon_observations: int = HORIZON_OBSERVATIONS,
) -> pd.Series:
    """Return ex-post annualized realized volatility for each return date.

    For a date ``t``, realized volatility is computed from the next
    ``horizon_observations`` trading returns.  Incomplete horizons are
    returned as ``NaN``.
    """
    if returns is None:
        _, returns = load_clean_data()

    daily_returns = returns.set_index("date")["return"].sort_index()
    result = pd.Series(np.nan, index=daily_returns.index, name="actual_rv")

    for position, date in enumerate(daily_returns.index):
        window = daily_returns.iloc[
            position + 1 : position + 1 + horizon_observations
        ].dropna()
        if len(window) == horizon_observations:
            result.loc[date] = np.sqrt(ANNUALIZATION * window.pow(2).mean())

    return result


if __name__ == "__main__":
    print(compute_actual_rv().dropna().tail())
