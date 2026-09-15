# %%
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOLATILITY_FILE = "flsssc4rjnjlhwsk.csv"
RETURNS_FILE = "spxReturn.csv"

def extract_volat():
    df = pd.read_csv(DATA_DIR / VOLATILITY_FILE, parse_dates=["date"])
    df = df[
        ((df["cp_flag"] == "P") & (df["delta"] == -25))
        | ((df["cp_flag"] == "C") & (df["delta"] == 50))
    ].copy()

    df = df.dropna()
    df = df.pivot_table(index="date", columns="cp_flag", values="impl_volatility")
    df["skew"] = df["P"] - df["C"]
    return df.sort_index()


def extract_ret():
    df = pd.read_csv(DATA_DIR / RETURNS_FILE, parse_dates=["date"])
    return df.set_index("date").sort_index()
