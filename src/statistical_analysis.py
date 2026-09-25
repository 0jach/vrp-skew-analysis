# %%
import numpy as np
import pandas as pd

from data_parser import RETURN_COLUMNS, VOLATILITY_COLUMNS, load_clean_data
from expected_rv import compute_expct_rv
from volatility import compute_actual_rv

volatility, returns = load_clean_data()

forecast = compute_expct_rv(returns).rename("rv_t+30").rename_axis("date").reset_index()
actual = compute_actual_rv(returns).rename("actual_rv_t+30").rename_axis("date").reset_index()

volatility = volatility.merge(
    forecast,
    on="date",
    how="left",
    validate="one_to_one",
)
volatility = volatility.merge(
    actual,
    on="date",
    how="left",
    validate="one_to_one",
)
# --- estimating true PDF f_x_z(x, z) where x = EV RP = IV^2 - RV_t+30 and z = skew

volatility["ev_rp"] = volatility["atm_iv_30d"] ** 2 - volatility["rv_t+30"] ** 2
print(volatility, "\n", np.mean(volatility["ev_rp"]))

# %%
import matplotlib.pyplot as plt

plt.hist(volatility["ev_rp"][volatility["ev_rp"] > -0.1], bins = 100)
plt.show()

# %%
volatility["ev_rp"].describe(percentiles=[
    .001, .005, .01, .05, .25, .5, .75, .95, .99, .995, .999
])
cols = ["date", "ev_rp", "atm_iv_30d", "rv_t+30"]
volatility.nsmallest(30, "ev_rp")[cols]

plt.plot(volatility["date"], volatility["ev_rp"])
plt.ylabel("EV RP")
plt.xlabel("Date")
plt.plot(volatility["date"], volatility["rv_t+30"])
plt.show()

# interpretation: period of crisis like 2008 can lead to expected rv being extremely higher that IV, so that EVPR is negative up to less that 100%.
# HAR as a predictor tends to predict high volatility for a long time after a shock / crisis. this makes the EVPR go negative.
# We can analyse the error distrubutions given skew.
# %%
har_error = (volatility[["date", "rv_t+30", "actual_rv_t+30", "skew"]].dropna().set_index("date").copy())
har_error["moving_mean_skew"] = har_error["skew"].rolling(
    "30D", closed="left"
).mean()

plt.plot(har_error.index, har_error["moving_mean_skew"])
plt.show()


# %%
har_error["error"] = har_error["rv_t+30"] - har_error["actual_rv_t+30"]
har_error["skew_group"] = pd.qcut(
    har_error["skew"],
    q=3,
    labels=["Low skew", "Medium skew", "High skew"],
)

low = har_error.loc[har_error["skew_group"] == "Low skew"]
medium = har_error.loc[har_error["skew_group"] == "Medium skew"]
high = har_error.loc[har_error["skew_group"] == "High skew"]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.boxplot(
    [low["error"], medium["error"], high["error"]],
    tick_labels=["Low", "Medium", "High"],
    showfliers=False,
)
ax.axhline(0, ls="--", lw=1, color="black")
ax.set_xlabel("Downside-skew regime")
ax.set_ylabel(r"HAR forecast error $\hat{RV}_{t,t+22}-RV_{t,t+22}$")
ax.set_title("HAR forecast errors across downside-skew regimes")
plt.tight_layout()
plt.show()

# %%
actual = (
    compute_actual_rv(returns)
    .rename_axis("date")
    .rename("actual_rv")
    .reset_index()
)
plot_data = volatility.merge(actual, on="date", how="left").dropna(
    subset=["atm_iv_30d", "rv_t+30", "actual_rv", "skew", "ev_rp"]
)
plot_data["point_size"] = 20 + 180 * plot_data["ev_rp"].abs()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for axis, realized, title in zip(
    axes,
    ["rv_t+30", "actual_rv"],
    ["Implied vs expected RV", "Implied vs actual RV"],
):
    scatter = axis.scatter(
        plot_data["atm_iv_30d"],
        plot_data[realized],
        c=plot_data["skew"],
        s=plot_data["point_size"],
        alpha=0.65,
        cmap="coolwarm",
    )
    limits = [0, max(plot_data["atm_iv_30d"].max(), plot_data[realized].max())]
    axis.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("30-day ATM implied volatility")
    axis.set_ylabel("Annualized realized volatility")
    axis.grid(alpha=0.25)

fig.colorbar(scatter, ax=axes, label="Downside skew")
fig.suptitle("Volatility risk-premium outliers")
plt.show()
