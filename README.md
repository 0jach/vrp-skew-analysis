# SPX volatility risk premium and downside skew

This repository contains an exploratory study of whether downside implied-volatility skew helps explain the relationship between expected and realized SPX volatility.

The current implementation works with daily SPX observations and is still research code. It builds a 30-day implied-volatility/skew dataset, estimates point-in-time realized-volatility forecasts, and compares forecast errors across skew regimes.

## What is currently implemented

For each date with the required option-surface nodes, the project extracts:

```text
30-day ATM IV  = 30-day 50-delta call implied volatility
downside skew  = 30-day 25-delta put IV - 30-day 50-delta call IV
```

Realized volatility is annualized with 252 trading days. The forward realized-volatility label uses the next 22 trading returns, so the window is `(t, t+22 trading days]`.

Two expanding, point-in-time forecast models are available:

- `HAR`: daily, 5-day, and 22-day realized-variance features;
- `HAR-X`: the same features plus contemporaneous downside skew.

The model target is the next 22-day realized variance. Forecasts are converted back to volatility, and training only uses targets whose full forward window is already known. `HAR_X_comparison.py` evaluates both models overall and within low-, medium-, and high-skew terciles using bias, mean absolute error, and root mean squared error.

`statistical_analisys.py` contains the current exploratory analysis. It also calculates the forecast-based quantity

```text
EVRP = 30-day ATM IV² - expected realized volatility²
```

and produces exploratory plots comparing implied, expected, and subsequently realized volatility.

## Input data

Place the following files in `data/`:

| File | Required columns |
| --- | --- |
| `flsssc4rjnjlhwsk.csv` | `secid`, `date`, `days`, `delta`, `impl_volatility`, `cp_flag` |
| `spxReturn.csv` | `secid`, `date`, `close`, `return` |

The option surface must contain exact rows for `days == 30`, a call with `delta == 50`, and a put with `delta == -25`. Dates without both option observations are dropped. The loader currently expects one unique observation per security/date/node and does not interpolate missing surface points.

The raw data files and generated outputs are ignored by Git. The local sample data covers one SPX security from 2005-01-03 through 2025-08-29, but the analysis code does not download data or validate vendor-specific conventions beyond the required columns and node values.

## Setup and usage

The project does not currently define an installable package or a test suite. Create an environment and install the three runtime dependencies:

```bash
python -m venv .venv
.venv/bin/pip install numpy pandas matplotlib
```

Run the main comparison from the repository root:

```bash
PYTHONPATH=src .venv/bin/python src/HAR_X_comparison.py
```

This prints overall and skew-tercile forecast metrics and writes:

```text
output/har_x_comparison_by_skew.png
```

Useful exploratory entry points are:

```bash
PYTHONPATH=src .venv/bin/python src/data_parser.py
PYTHONPATH=src .venv/bin/python src/expected_rv.py
PYTHONPATH=src .venv/bin/python src/volatility.py
PYTHONPATH=src .venv/bin/python src/statistical_analisys.py
```

The last command opens interactive Matplotlib figures and is intended for exploratory use. The default command-line comparison uses `min_training=2520` observations; the forecast functions default to 252 observations when called directly.

## Repository layout

```text
data/
  flsssc4rjnjlhwsk.csv   raw option surface
  spxReturn.csv          SPX returns
output/                  generated figures
src/
  data_parser.py         input loading, validation, and IV-node extraction
  volatility.py          forward realized-volatility calculation
  expected_rv.py         expanding HAR forecasts
  HAR_X.py               expanding HAR-X forecasts
  HAR_X_comparison.py    HAR versus HAR-X evaluation and plotting
  statistical_analisys.py exploratory EVRP plots and diagnostics
```

## Limitations and next steps

This is not yet a trading strategy or a production research pipeline. It does not include option prices, delta-hedged returns, transaction costs, formal statistical inference, automated tests, or a walk-forward strategy backtest. The next steps are to formalize the data pipeline, validate the forecast specification, add robust statistical tests, and evaluate whether skew-conditioned signals survive realistic trading costs.

## License

The project is released under the Apache License 2.0; see [LICENSE](LICENSE).
