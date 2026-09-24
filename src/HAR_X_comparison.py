"""Compare standalone HAR and HAR-X forecasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_parser import load_clean_data
from expected_rv import compute_expct_rv
from HAR_X import compute_har_x_rv
from volatility import compute_actual_rv


def plot_by_skew(
    by_skew: pd.DataFrame,
    output_path: str | Path = "output/har_x_comparison_by_skew.png",
    show: bool = False,
) -> Path:
    """Plot HAR and HAR-X forecast errors across ordered skew terciles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = [str(group) for group in by_skew.index]
    positions = np.arange(len(groups))
    width = 0.34
    colors = {"HAR": "#4C78A8", "HAR-X": "#F58518"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    for axis, metric, title in zip(
        axes[:2],
        ("mae", "rmse"),
        ("Mean absolute error", "Root mean squared error"),
    ):
        axis.bar(
            positions - width / 2,
            by_skew[f"har_{metric}"],
            width,
            label="HAR",
            color=colors["HAR"],
        )
        axis.bar(
            positions + width / 2,
            by_skew[f"har_x_{metric}"],
            width,
            label="HAR-X",
            color=colors["HAR-X"],
        )
        axis.set_title(title)
        axis.set_ylabel("Volatility forecast error")
        axis.grid(axis="y", alpha=0.25)

    bias_axis = axes[2]
    bias_axis.errorbar(
        positions - 0.08,
        by_skew["har_bias"],
        yerr=by_skew["har_std"],
        fmt="o-",
        capsize=4,
        linewidth=2,
        label="HAR",
        color=colors["HAR"],
    )
    bias_axis.errorbar(
        positions + 0.08,
        by_skew["har_x_bias"],
        yerr=by_skew["har_x_std"],
        fmt="o-",
        capsize=4,
        linewidth=2,
        label="HAR-X",
        color=colors["HAR-X"],
    )
    bias_axis.axhline(0, color="black", linewidth=1, linestyle="--")
    bias_axis.set_title("Bias with ±1 error SD")
    bias_axis.set_ylabel("Signed forecast error")
    bias_axis.grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.set_xticks(positions, groups)
        axis.set_xlabel("Skew tercile")
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def compare_har_models(
    volatility: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    min_training: int = 252,
    plot_path: str | Path = "output/har_x_comparison_by_skew.png",
    show_plot: bool = False,
) -> pd.DataFrame:
    if volatility is None or returns is None:
        volatility, returns = load_clean_data()

    forecasts = pd.DataFrame(
        {
            "har": compute_expct_rv(returns, min_training),
            "har_x": compute_har_x_rv(
                returns, volatility[["date", "skew"]], min_training
            ),
            "actual": compute_actual_rv(returns),
            "skew": volatility.set_index("date")["skew"],
        }
    ).dropna()

    errors = forecasts[["har", "har_x"]].sub(forecasts["actual"], axis=0)
    overall = pd.DataFrame(
        {
            "nobs": errors.count(),
            "bias": errors.mean(),
            "mae": errors.abs().mean(),
            "rmse": (errors.pow(2).mean()) ** 0.5,
        }
    )

    errors["skew_group"] = pd.qcut(
        forecasts["skew"],
        q=3,
        labels=["Low skew", "Medium skew", "High skew"],
    )
    errors["har_error"] = errors["har"]
    errors["har_x_error"] = errors["har_x"]
    by_skew = errors.groupby("skew_group", observed=True).agg(
        har_mae=("har_error", lambda x: x.abs().mean()),
        har_rmse=("har_error", lambda x: (x.pow(2).mean()) ** 0.5),
        har_bias=("har_error", "mean"),
        har_std=("har_error", "std"),
        har_x_mae=("har_x_error", lambda x: x.abs().mean()),
        har_x_rmse=("har_x_error", lambda x: (x.pow(2).mean()) ** 0.5),
        har_x_bias=("har_x_error", "mean"),
        har_x_std=("har_x_error", "std"),
    )

    print("Overall:")
    print(overall)
    print("\nBy skew group:")
    print(by_skew)
    saved_plot = plot_by_skew(by_skew, plot_path, show=show_plot)
    print(f"\nSaved skew comparison plot to {saved_plot}")
    return overall


if __name__ == "__main__":
    compare_har_models(min_training=2520)
