# Where Did the Edge Go?

Code for the working paper *Where Did the Edge Go? Leakage, Naive Baselines and Non-Stationary Features in a Machine-Learning Trading Signal* (Mohammed Hadi, November 2026).

The study re-creates QuantAI's original prediction pipeline (the features, XGBoost + LSTM ensemble and 5-day label as they were before the fix in [#1](https://github.com/mhdhadi2005/QuantAI/pull/1)) and scores it three ways on AAPL, MSFT, NVDA, SPY and QQQ:

| Protocol | What it does | Ensemble edge over "always up" |
|---|---|---|
| P0 in-sample | train and score on the same window (what the old AI backtest did) | +23 to +35 pp |
| P1 80/20 split | the original validation, no purge | −16 to −2 pp |
| P2 purged walk-forward | retrain yearly on earlier data only, 2016–2026 | −10 to −4 pp |

It also runs an ablation without raw price-level features, and a long/flat strategy with a one-bar delay and costs, compared against buy-and-hold.

## Reproduce

```bash
pip install pandas numpy scikit-learn xgboost torch yfinance matplotlib
python experiment.py      # ~4 min; downloads Yahoo Finance data, writes results/ and figures/
python build_paper.py     # writes paper.html with every number filled from results/results.json
```

Seeds are fixed. A repeated run reproduces every reported number; the raw outputs differ only by floating-point round-off from re-reading cached prices. Print `paper.html` to PDF from any browser.

## Files

- `experiment.py`: data download, features, models, the three protocols, bootstrap intervals, the strategy test and both figures
- `build_paper.py`: builds the paper from `results/results.json`
- `results/results.json`: the results reported in the paper
- `figures/`: Figure 1 (edge by protocol) and Figure 2 (equity curves)

Price data is not committed: `experiment.py` downloads it and caches it in `results/` (git-ignored).
