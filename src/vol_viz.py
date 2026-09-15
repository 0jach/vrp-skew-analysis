# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/volatility_full.csv") # another volatility file to visualize how IV changes with time and delta
df = df.dropna(subset=["days", "delta", "impl_volatility"])
df = df[df["cp_flag"] == "P"]

surface = df.pivot_table(
    index="days",
    columns="delta",
    values="impl_volatility",
    aggfunc="mean",
).sort_index().sort_index(axis=1)

surface = surface.dropna()
# %%
X, Y = np.meshgrid(surface.columns, surface.index)
Z = surface.to_numpy()
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(azim=135)

ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("Days to expiration")
ax.set_ylabel("Delta")
ax.set_zlabel("Implied volatility")

plt.show()

# %%
# %%
from data_parser import extract_volat


def plot_volatility_and_skew(df):
    _fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    df["C"].plot(ax=axes[0], title="ATM Implied Volatility")
    axes[0].set_ylabel("IV")

    df["skew"].plot(ax=axes[1], title="Downside Skew")
    axes[1].set_ylabel("Put IV - ATM IV")

    plt.tight_layout()
    plt.show()

plot_volatility_and_skew(extract_volat())
