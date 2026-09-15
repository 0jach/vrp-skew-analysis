# VRP and Skew Analysis

Work in progress on when selling SPX volatility is attractive, and when a high volatility risk premium is mainly compensation for downside tail risk.

## Current state

The project currently includes:

- basic parsing and filtering of SPX option and return data;
- 30-day ATM implied volatility and 25-delta put skew measures;
- weekly, forward 30-day realized-volatility labels;
- historical 14-, 30-, and 60-day realized-volatility features;
- a weekly modelling dataset joining the option data, features, and target;
- exploratory plots of implied volatility and skew.

The data pipeline and methodology are still being developed, so the current outputs should not be treated as final research results.

## Next step

Use past 30-day realized volatility as the first forecast of future 30-day realized volatility. Evaluate it with MAE and RMSE, then use the forecast to calculate the ex-ante volatility risk premium.

## Final goal

The goal is to estimate the ex-ante volatility risk premium using a real-time forecast of future realized volatility, then test whether downside skew helps distinguish attractive short-volatility opportunities from regimes with elevated crash risk.

The finished project should include walk-forward volatility forecasts, VRP/skew regime analysis, and a transaction-cost-aware SPX option backtest, with emphasis on both returns and tail risk.
