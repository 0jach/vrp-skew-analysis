"""Import the unprocessed option-surface and SPX return data."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOLATILITY_FILE = "flsssc4rjnjlhwsk.csv"
RETURNS_FILE = "spxReturn.csv"

VOLATILITY_COLUMNS = {
    "secid",
    "date",
    "days",
    "delta",
    "impl_volatility",
    "cp_flag",
}
RETURN_COLUMNS = {"secid", "date", "close", "return"}


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Read a raw CSV and fail clearly when its schema is incomplete."""
    if not path.is_file():
        raise FileNotFoundError(f"Unprocessed data file not found: {path}")

    data = pd.read_csv(path, parse_dates=["date"])
    missing = required_columns.difference(data.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required columns: {columns}")
    return data


def load_unprocessed_data(
    data_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw volatility surface and SPX returns as data frames."""
    return load_volatility_data(data_dir), load_return_data(data_dir)


def load_volatility_data(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Import the unprocessed volatility-surface CSV."""
    return _read_csv(Path(data_dir) / VOLATILITY_FILE, VOLATILITY_COLUMNS)


def load_return_data(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Import the unprocessed SPX return CSV."""
    return _read_csv(Path(data_dir) / RETURNS_FILE, RETURN_COLUMNS)


def clean_volatility_data(volatility: pd.DataFrame) -> pd.DataFrame:
    """Return one observation per security-date with one column per IV measure.

    The raw surface stores the call/put node identity across ``cp_flag`` and
    ``delta``.  For this analysis, the 30-day 50-delta call and 25-delta put
    are separate measured variables, so they are pivoted into explicit
    columns.  Dates without both measurements are incomplete observations and
    are removed.
    """
    missing = VOLATILITY_COLUMNS.difference(volatility.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Volatility data is missing required columns: {columns}")

    nodes = volatility.loc[
        (volatility["days"] == 30)
        & (
            ((volatility["cp_flag"] == "P") & (volatility["delta"] == -25))
            | ((volatility["cp_flag"] == "C") & (volatility["delta"] == 50))
        ),
        ["secid", "date", "cp_flag", "impl_volatility"],
    ]

    duplicate_keys = nodes.duplicated(["secid", "date", "cp_flag"], keep=False)
    if duplicate_keys.any():
        raise ValueError("Volatility data has duplicate security-date-IV nodes")

    tidy = (
        nodes.pivot(
            index=["secid", "date"],
            columns="cp_flag",
            values="impl_volatility",
        )
        .rename(columns={"C": "atm_iv_30d", "P": "put_25d_iv_30d"})
        .rename_axis(columns=None)
        .dropna(subset=["atm_iv_30d", "put_25d_iv_30d"])
        .reset_index()
        .sort_values(["date", "secid"], ignore_index=True)
    )
    tidy["skew"] = tidy["put_25d_iv_30d"] - tidy["atm_iv_30d"]
    return tidy


def clean_return_data(returns: pd.DataFrame) -> pd.DataFrame:
    """Return one complete SPX market observation per security-date."""
    missing = RETURN_COLUMNS.difference(returns.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Return data is missing required columns: {columns}")

    tidy = returns.loc[:, ["secid", "date", "close", "return"]].copy()
    tidy = tidy.dropna(subset=["secid", "date", "close", "return"])
    if tidy.duplicated(["secid", "date"]).any():
        raise ValueError("Return data has duplicate security-date observations")
    return tidy.sort_values(["date", "secid"], ignore_index=True)


def load_clean_data(
    data_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both raw inputs into tidy analysis tables."""
    volatility, returns = load_unprocessed_data(data_dir)
    return clean_volatility_data(volatility), clean_return_data(returns)


def extract_volat(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Return the cleaned volatility data in the legacy date-indexed format."""
    tidy = clean_volatility_data(load_volatility_data(data_dir))
    return (
        tidy.drop(columns="secid")
        .rename(columns={"atm_iv_30d": "C", "put_25d_iv_30d": "P"})
        .set_index("date")
    )


def extract_ret(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Return date-indexed, chronologically sorted SPX market data."""
    return clean_return_data(load_return_data(data_dir)).set_index("date")


if __name__ == "__main__":
    volatility, returns = load_clean_data()
    print(f"Loaded {len(volatility):,} clean volatility observations")
    print(f"Loaded {len(returns):,} clean return observations")
