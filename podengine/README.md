# Pod Engine

A standalone, multi-strategy portfolio engine run the way a multi-manager fund runs its book: several
independent strategies ("pods"), a central allocator that funds them according to their out-of-sample
record, mechanical risk limits, and a validation gate that tries hard to prove the results are luck.

Long, short, leveraged and futures-ready. Free data (Yahoo Finance), no API keys, no other projects needed.

## Quick start

```bash
pip install numpy pandas scipy yfinance          # Python 3.10+
python -m podengine run --demo                   # offline demo on simulated data (about 75 s)
python -m podengine run                          # real prices, full report with robustness grid
python -m podengine today --capital 250000       # fast daily update: today's targets and order sheet
python -m podengine optimize --positions examples/positions_example.csv --views examples/views_example.json
                                                 # risk report and better weights for what you already hold
python -m podengine optimize --demo              # optimiser demo, offline
```

Open `output/report.html` in a browser. The same folder gets `orders.csv`, `target_weights.csv`,
`daily_returns.csv`, `pod_returns.csv`, `ablation.csv` and `summary.json`.

Useful options:

| Option | What it does |
|---|---|
| `--profile unconstrained` | Futures / portfolio margin: long and short, up to 6x gross (default) |
| `--profile margin` | Reg-T style: long and short, 2x gross, higher financing cost |
| `--profile cash` | Long only, no leverage, 40% max per asset |
| `--vol-target 0.15` | Portfolio volatility target (default 12%) |
| `--capital 50000` | Account size for the order sheet |
| `--positions held.csv` | Your current holdings (`ticker,shares`) so the order sheet shows the trades |
| `--tickers SPY,TLT,GLD,...` | Your own Yahoo universe instead of the default 42 ETFs |
| `--prices my.csv --meta meta.json` | Your own price file (date column, one column per ticker) |
| `--config my.json` | Override any setting in `podengine/config.py` |
| `--quick` | Skip the robustness grid (`today` always does) |
| `--no-futures` | Don't look up futures prices |

Run the tests with `python -m pytest -q` (55 tests, about 90 s). They include automated look-ahead tests: the
engine is re-run on data cut at several past dates and every decision must be identical, and a deliberately
cheating signal (peeking two days ahead) must be caught.

## How it works

```
VALIDATION GATE  Deflated Sharpe vs every configuration ever tried · overfitting probability · bootstrap interval
      │
PODS  trend │ value + momentum │ defensive (low beta) │ risk parity │ reversal (experimental)
      │      each pod is sized to 10% stand-alone volatility and keeps its own paper P&L
      │
ALLOCATOR  regime-aware covariance → risk budget per pod ∝ shrunk out-of-sample Sharpe → half-Kelly cap
      │
GOVERNOR   pod down 1.5× its vol → risk halved · down 2.25× → off until it recovers
           turbulence (unusual cross-asset moves) → de-risk · tail-loss cap · leverage and position caps
      │
EXECUTION  trade one day after the signal · no-trade band · costs, financing and short fees charged
      │
ANALYTICS  regime Monte Carlo with self-correcting ranges · risk by asset class and pod · stress episodes · order sheet
```

### Where each idea comes from

| Layer | Source | What it does here |
|---|---|---|
| Many small, uncorrelated edges | Renaissance; Grinold's law IR ≈ IC·√breadth | Five pods across ~40 assets in five asset classes instead of one big forecast |
| Trend | AQR, managed-futures funds | Time-series momentum over 1, 3, 6 and 12 months, volatility-scaled, long and short |
| Value + momentum | Asness, Moskowitz & Pedersen (2013) | Cross-sectional ranks within peer groups; the two signals are negatively correlated, so they cover each other |
| Defensive | Frazzini & Pedersen (2014), betting against beta | Long low-beta, short high-beta equity markets, beta-neutral |
| Risk parity | Bridgewater All Weather | Equal risk from equities, government bonds, credit and real assets |
| Pod risk limits | Multi-manager platforms | Mechanical halving and switch-off on drawdown |
| Sizing | Thorp / Kelly | Volatility target capped at half the growth-optimal level implied by the evidence |
| Overfitting control | Bailey & López de Prado | Deflated Sharpe with a persistent trial log; probability of backtest overfitting |

### The quant edges added on top

* **Volatility models compete.** A HAR cascade model (daily, weekly, monthly and yearly volatility feeding
  tomorrow's, the market version of a turbulent energy cascade) races an EWMA baseline. Both are scored out
  of sample with QLIKE loss and blended toward whichever has been more accurate. Open/high/low data feeds a
  range-based estimator that is several times less noisy than close-to-close returns.
* **Covariance cleaned with random-matrix theory.** Correlation eigenvalues inside the Marchenko-Pastur noise
  band are flattened before any optimisation. The result is then blended toward the correlations seen in past
  stress periods, in proportion to the hidden-Markov model's probability of stress tomorrow.
* **The risk model checks itself.** Realised versus predicted volatility is tracked for every pod and for the
  whole portfolio, and sizes are rescaled when the forecasts have been running too low or too high.
* **Pod capital follows evidence, not hope.** Each pod's Sharpe is a Bayesian blend of a sceptical prior
  (about half the published figure) and its own live record, using Lo's (2002) standard error. Experimental
  ideas start at a prior of zero.
* **Ranges that are honest by construction.** The Monte Carlo re-plays real historical days drawn from the
  current regime, with volatility clustering. Adaptive conformal inference then audits its 90% and 50% ranges
  at every past date and adjusts the quantile levels after each miss, so long-run coverage converges to the
  stated level. That convergence is a theorem about the procedure, not an assumption about markets.
* **Fat-tail Kelly.** Growth-optimal leverage is found by golden-section search on the simulated return
  distribution, not from μ/σ², so crash risk lowers it.
* **Cost discipline.** Signals trade a day late, rank scores are smoothed, and a no-trade band skips small
  rebalances. The ablation table shows the effect of each.

## Reading the report

1. **Today's portfolio:** target weights, leverage, regime, and the self-corrected 1-week and 1-month ranges.
2. **Pod scoreboard:** raw versus shrunk Sharpe, drawdown status (full / halved / off / benched), and each
   pod's contribution.
3. **Forecast ranges:** how often past 90% and 50% ranges actually contained the outcome, before and after
   correction, plus the one-year fan chart and the Kelly check.
4. **Backtest:** against stocks and 60/40, with calendar years, stress episodes and the worst drawdowns.
5. **Validation gate:** Deflated Sharpe, overfitting probability, and an ablation table where each row removes
   one layer. If removing a layer helps on your data, that layer is not earning its place there.
6. **Order sheet:** ETF shares, and where a liquid contract exists, the equivalent futures (E-mini or micro,
   Treasury, gold, crude, FX) with the rounding error shown.

## Value at Risk

Every engine report has a VaR section, and the optimiser computes the same numbers for your own holdings.

* **Today:** 1-day VaR at 95% and 99%, and 1-day Expected Shortfall at 97.5%, by three methods:
  * normal (bell curve on the engine's covariance);
  * filtered historical simulation (the last 1,000 days of real joint moves, each rescaled to today's volatility);
  * the regime Monte Carlo, which also gives 10-day figures. Volatility clusters, so √10 scaling understates the 10-day risk.
* **Attribution:** ES is split across assets and across pods (Euler allocation, so the parts add up to the total).
* **Backtest:** every past day's 1-day VaR, made with data up to the previous close, is compared with the P&L
  that followed. The report shows:
  * Kupiec's test (are breaches as frequent as stated?);
  * Christoffersen's test (do they cluster?);
  * the Basel traffic light (99% breaches in the last 250 days: 0-4 green, 5-9 yellow, 10+ red);
  * the Acerbi-Székely Z2 test of ES, with a p-value simulated from the model's own scenarios.

  A self-corrected variant adjusts its quantile after every breach, like the Monte Carlo ranges do.

VaR and ES here cover stocks, ETFs and futures-equivalent exposure. Options are not handled.

## Optimising a portfolio you hold

`python -m podengine optimize --positions holdings.csv` reads your positions and proposes three
alternatives next to your current mix, using the same risk model as the engine.

**Positions file:** a `ticker` column plus `shares`, `value` (dollars) or `weight`. Add a `CASH` row for cash;
use negative numbers for shorts. See `examples/positions_example.csv`.

| Alternative | What it does | Needs return forecasts? |
|---|---|---|
| Minimum variance | Lowest-risk mix of your assets at the same net exposure | No |
| Risk parity | Every holding contributes equal risk, long only, scaled to the target volatility | No |
| Black-Litterman | Highest expected return at the target volatility, net of trading costs | Yes (see below) |

**How Black-Litterman gets its expected returns.** It starts from the returns that would make your current
portfolio optimal, at a Sharpe ratio of 0.30 (`--prior-sharpe`). With no views, the answer is therefore your
current portfolio, and a test checks this. Views then move it as far as their confidence allows. Hedges such as
bonds in an equity-heavy book get low or negative starting returns, because that is what justifies holding them
at their current size; it is not a forecast that they will fall.

**Views file** (`--views`), a JSON list:

```json
[{"label": "Long bonds beat cash by 4%", "assets": {"TLT": 1}, "return": 0.04, "confidence": 0.6},
 {"label": "Nasdaq lags the S&P by 3%", "assets": {"QQQ": 1, "SPY": -1}, "return": -0.03, "confidence": 0.5}]
```

* One asset makes an absolute view; +1/-1 makes a relative one.
* `return` is the annual return over cash.
* `confidence` 0.5 gives the view the same weight as the starting point, which puts the blend halfway between
  them (also tested). Higher confidence pulls the blend closer to the view. This mapping is a common heuristic,
  not a law.
* The prior's uncertainty is τ = 0.05 (`--tau`).

**Other options:**

* `--vol-target`: target volatility (default: your current volatility, for a like-for-like comparison).
* `--max-turnover 0.3`: caps how much weight Black-Litterman may trade. Strong views can otherwise move it a lot.
* `--add XLE,EFA`: tickers it may buy that you don't hold.
* `--profile`: sets shorting and leverage, as for the engine.

Holdings with less than a year of history are kept at their current weight and left out of the optimisation.
Tickers with no data are reported.

**Output:** `output/optimizer.html` and `output/optimizer_trades.csv` (share trades for every alternative).

## Animated viewer (one page, tabs)

```bash
python -m podengine run --viz          # full run, then export the viewer data (or: today --viz)
python -m podengine viz --demo         # backtest-only export, offline demo (about 1.5 minutes)
python -m podengine optimize --positions holdings.csv --viz   # also puts the optimizer report in the viewer
python -m podengine serve              # opens http://127.0.0.1:8765/
```

The `viz/` folder is a self-contained page with seven tabs: the engine report and the optimizer report
(embedded as they are), plus five animated views. It ships with demo data, so `serve` works straight away.
Browsers block scripts and data loaded from `file://`, so it has to be served; opened as a file it says so.

* **Python does all the maths.** `podengine/vizexport.py` writes precomputed frames to `viz/data/*.json`,
  defined by `viz/data/schema.json`. The browser only draws, on plain canvas, with no external chart libraries.
* **Animations toggle, off by default.** Off means no render loop, and the animation module (`js/anim.js`) is
  never downloaded. The charts are still drawn as static pictures, and the date slider still works. The choice
  is saved in the browser (localStorage, with a fallback if storage is blocked).
* **Controls on every view:** play/pause, speed (0.5× to 4×) and a date slider. Playback pauses when the chart
  scrolls off-screen or the browser tab is hidden, and resumes when it is visible again.
* **No effect on the engine:** the export only reads results and uses its own random seed. Without `--viz`,
  no viewer code is even imported.

| View | What it shows |
|---|---|
| Forecast fan | Monte Carlo 5/25/50/75/95 percentile bands for the engine's target weights over the next 21 days, with the realised path drawn through them. Replay reveals each month's outcome, then re-forecasts at the next date. |
| Calibration | Histogram of where each outcome landed within its forecast distribution (PIT), against the uniform line. It flags tails that are too thin or too fat (binomial test on the outer 10% at each end, p < 0.05, from 30 forecasts on), and builds up as the replay advances. |
| Regimes | Ridgeline of the portfolio's 21-day return density for a start in each regime (10,000 paths per ridge, smoothed density), with the current regime in mint and the unconditional distribution as a dashed gold line. It moves through history as regime probabilities shift. |
| Frontier | Rolling efficient frontier from the trailing year: Ledoit-Wolf covariance, Bayes-Stein shrunk means, and the engine's leverage, per-asset and shorting limits. The engine's portfolio is a moving dot with a fading trail. |
| Exposure | Long and short weights by asset class stacked above and below zero, gross (= leverage) and net lines, and the gross cap. |

Labels on each view say "Simulated / historical replay · not a forecast" and whether figures are gross of costs.
All viewer figures are gross of trading costs and financing; the engine report has the net figures.

Design choices worth knowing:

* **Forecast dates:** one every 21 trading days, so each outcome is scored once. Re-forecasting at every weekly
  rebalance would overlap the horizons and count the same market move several times in the calibration histogram.
* **Fan and calibration bands:** these are the Monte Carlo's raw ranges, before the engine's self-correction,
  so the histogram shows how good the raw model is.
* **Regime ridges:** a ridge means "start in this regime, then switch regimes as the fitted model allows".
  "Unconditional" starts from the model's long-run mix of regimes.
* **Frontier expected returns:** raw trailing averages produced absurd frontiers (in-sample Sharpe ratios near
  7), so the means are shrunk with Bayes-Stein (Jorion 1986). The frontier is still in-sample and backward-looking,
  an upper bound the real portfolio should not be expected to reach. With leverage allowed, its lower part is a
  straight line from cash; the best-Sharpe marker sits where that line ends.
* **Exposure:** weights are grouped by asset class, because 40-odd individual assets would be unreadable stacked.
  The exported positions themselves are per asset in `daily_returns.csv` and `target_weights.csv`.

Tests (`tests/test_viz.py`):

* The PIT comes out uniform when the forecast model is known to be correct, for normal and fat-tailed cases,
  and it flags forecasts that are too narrow or too wide.
* Ledoit-Wolf matches scikit-learn to machine precision.
* The frontier respects the engine's limits. Under shrinkage it moves less as the window rolls, and its true risk
  is closer to what it promised; the gain grows with the number of assets per observation.
* Every export validates against the schema, both with the `jsonschema` package and with the built-in validator.
* Engine results are identical with and without an export.
* In a real headless Chrome:
  * with animations off: no animation frames, no animation code fetched, no global variables;
  * with them on: the loop runs, pauses off-screen and resumes on-screen;
  * the `file://` warning shows.

  The browser tests need Chrome or Chromium and the `websockets` package, and are skipped otherwise.

## Futures

Research runs on ETFs because they have clean, adjusted, free history. Execution can use futures:
`universe.FUTURES` maps each ETF to its contracts and multipliers. Equity and commodity hedge ratios are
re-estimated from a year of data. For Treasuries the table holds approximate duration ratios (ZF ≈ 1.05,
ZN ≈ 1.20, ZB ≈ 1.35 per $1 of ETF); check them against your broker's DV01 before trading size.
Futures roll costs are not modelled; ETF expense ratios are already inside the prices.

## Honest limits

* No backtest is a forecast. Expected returns are barely predictable, so the engine never depends on a point
  forecast of direction; its strength is in sizing, diversification and risk control.
* Free data covers funds and stocks that exist today, so a stock universe will be flattered by
  survivorship. The default ETF universe is far less exposed to this.
* Trading costs are flat per asset class (2-5 bp one way). Real costs widen in stress; use `cost_mult`.
* Value is a price-only proxy (cheap versus five years ago), not book value.
* Regime probabilities lag turning points. Risk parity assumes stocks and bonds do not fall together for long,
  and 2022 is the counterexample.
* Every configuration you try is logged in `state/trials.jsonl` and raises the Deflated Sharpe hurdle. Tuning
  until the backtest looks good is exactly what the gate is built to catch.

This is a research tool, not investment advice.

## Extending it

Add a pod by subclassing `pods.Pod` (implement `prepare` and `weights`, using only data up to the current row),
registering it in `pods.REGISTRY`, and adding a prior in `config.py`. Its prior should be at most half the
published Sharpe ratio. The look-ahead test will tell you if the new signal peeks at the future.

## Files

```
podengine/
  universe.py     default 42-ETF universe, asset classes, cost assumptions, futures specs
  config.py       every setting and the three account profiles
  data.py         Yahoo loader with cache, CSV loader, synthetic demo data
  vol.py          HAR vs EWMA volatility with out-of-sample blending
  cov.py          correlation cleaning (random-matrix clipping, shrinkage), Euler risk contributions
  regime.py       hidden-Markov regimes (walk-forward, filtered), turbulence index
  pods/           the five strategies
  allocator.py    risk budgeting, Bayesian Sharpe, drawdown governor, limits, golden-section search
  backtest.py     walk-forward engine (same code path produces today's targets)
  var.py          VaR/ES by three methods, Euler attribution, Kupiec/Christoffersen/Basel/Z2 backtests
  optimizer.py    optimise your own holdings: minimum variance, risk parity, Black-Litterman
  vizexport.py    precomputes every frame of the animated viewer (fan, PIT, regimes, frontier, exposure)
  vizschema.py    validates the viewer data against viz/data/schema.json
viz/              the single-page viewer: index.html, viz.css, js/ (app, views, anim), data/, reports/
  montecarlo.py   regime-switching filtered historical simulation, conformal calibration, Kelly search
  validate.py     PSR, Deflated Sharpe, PBO, bootstrap, trial log, ablation grid
  analytics.py    performance, drawdowns, stress episodes, exposures, pod scoreboard
  execution.py    order sheet in ETF shares and futures contracts
  pipeline.py     runs everything
  report.py       self-contained HTML report
tests/            engine tests, VaR and optimiser tests
examples/         positions and views file examples
```

## Example reports

`docs/example_report_demo.html` comes from the simulated demo data, and `docs/example_optimizer_demo.html` is the
optimiser demo on the same data. `docs/example_report_20_stocks.html`
comes from a public file of 20 large US stocks (1990-2018), closes only, margin profile. That second universe
is all stocks that survived and became giants, so its Sharpe of about 1.0 is heavily flattered by survivorship.
It is there to show the report on real prices, not as evidence of an edge. Run the default ETF universe on
your own machine for a meaningful result.

## Review log

An independent reviewer audited the engine before release. It found no look-ahead bias: re-running the engine
on data cut at past dates reproduced every target, volatility forecast, regime probability and Monte Carlo
input to about 1e-15. The issues it did find were fixed:

* Weights now drift with prices between trades. Before the fix, cost and turnover were understated by about 17%.
* The Bayesian Sharpe update used the annual Sharpe where Lo's formula needs the per-period one, which
  over-shrank strong pods.
* Pod attribution now decomposes the P&L actually held; what account limits cause is shown separately.
* The Monte Carlo standardises history within each regime, so crisis-regime volatility is no longer counted twice.
* The margin profile no longer earns the cash rate on short proceeds (`short_rebate`).
* Pod paper returns now pay financing.
* The Deflated Sharpe headline uses the stricter of two hurdles.
* The turbulence index now warms up correctly.
* The shared risk-model cache now checks every setting it depends on.
* The look-ahead test now checks targets at several cut dates. A negative-control test proves it catches a
  signal that peeks just two days ahead.

Fixed while adding VaR and the optimiser:

* The covariance cache was keyed by date only. Asking for a different set of assets on the same date would have
  returned the wrong matrix. The engine never did this, but the optimiser would have.
