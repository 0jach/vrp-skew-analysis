# SPX volatility risk premium and downside skew

This project asks whether downside implied-volatility skew changes the relation
between the ex-ante volatility risk premium and subsequent volatility outcomes.

For every SPX date it constructs:

```text
skew = 30-day 25-delta put IV - 30-day 50-delta call IV
EVRP = ATM IV² - forecast realized volatility²
ex-post VRP = ATM IV² - subsequent realized volatility²
```

Volatilities use decimal units and 252-day annualization. Subsequent realized
volatility is measured over `(t, t+30 calendar days]`.

The pipeline compares historical RV, EWMA, and expanding HAR forecasts. The
main empirical specification deliberately uses trailing historical RV as the
simple forecast benchmark. It produces an independent 3×3 EVRP/skew sort and
interaction regressions with 22-lag Newey-West standard errors.

## Expected data

The project assumes these files and columns are already correct:

- `data/flsssc4rjnjlhwsk.csv`: `date`, `secid`, `days`, `delta`,
  `impl_volatility`, `cp_flag`;
- `data/spxReturn.csv`: `date`, `close`, `return`.

The option file must contain exact 30-day, 50-delta call and −25-delta put
surface nodes. There is intentionally no configurable schema or interpolation
layer.

The surface does not contain tradable option-price histories, so the project
does not calculate delta-hedged straddle returns or strategy performance.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python -m pytest
```

Generated datasets, tables, and figures are written under `data/processed/`
and `output/`.

## Code map

```text
src/vrp/data.py       fixed input loading and exact IV-node extraction
src/vrp/features.py   realized volatility, EVRP, and ex-post VRP
src/vrp/forecasts.py  historical, EWMA, and leakage-safe HAR forecasts
src/vrp/analysis.py   3×3 sorts and HAC interaction regressions
src/vrp/plots.py      the four research figures
src/vrp/pipeline.py   end-to-end orchestration and output writing
scripts/run_pipeline.py
tests/test_core.py
```
