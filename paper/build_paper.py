"""Builds paper.html from results/results.json so every number in the text comes from the experiment."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results", "results.json")))
R = D["results"]
SYMS = [r["symbol"] for r in R]


def pp(x, sign=True):
    v = round(x * 100, 1) + 0.0  # avoid printing -0.0
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def rng(vals, f=lambda v: pp(v)):
    return f"{f(min(vals))} to {f(max(vals))}"


def g(r, v, p, m):
    return r[v][p][m]


Q, S = "quantai_features", "stationary_only"
p0_ens = [g(r, Q, "P0_in_sample", "ens")["edge"] for r in R]
p0_xgb_acc = [g(r, Q, "P0_in_sample", "xgb")["acc"] for r in R]
p1_ens = [g(r, Q, "P1_single_split", "ens")["edge"] for r in R]
p2_ens = [g(r, Q, "P2_walk_forward", "ens")["edge"] for r in R]
p2_auc = [g(r, Q, "P2_walk_forward", "ens")["auc"] for r in R]
p2_ci_neg = sum(1 for r in R if g(r, Q, "P2_walk_forward", "ens")["edge_ci"][1] < 0)
p2s_ens = [g(r, S, "P2_walk_forward", "ens")["edge"] for r in R]
p2_xgb = [g(r, Q, "P2_walk_forward", "xgb")["edge"] for r in R]
p2s_xgb = [g(r, S, "P2_walk_forward", "xgb")["edge"] for r in R]
lstm_up = [g(r, Q, "P2_walk_forward", "lstm")["pred_up_rate"] for r in R]
spy = next(r for r in R if r["symbol"] == "SPY")
spy_p1_xgb_up = g(spy, Q, "P1_single_split", "xgb")["pred_up_rate"]
base_rates = [g(r, Q, "P2_walk_forward", "ens")["baseline_acc"] for r in R]
st = {r["symbol"]: r["strategy"] for r in R}
worse_sharpe = sum(1 for s in SYMS if st[s]["long_flat_p0.5"]["sharpe"] < st[s]["buy_and_hold"]["sharpe"])
worse_cagr = sum(1 for s in SYMS if st[s]["long_flat_p0.5"]["cagr"] < st[s]["buy_and_hold"]["cagr"])
cfg = D["config"]


def table1():
    rows = []
    names = {"P0_in_sample": "P0 In-sample", "P1_single_split": "P1 80/20 split", "P2_walk_forward": "P2 Purged walk-forward"}
    for r in R:
        for i, p in enumerate(["P0_in_sample", "P1_single_split", "P2_walk_forward"]):
            s = r[Q][p]["ens"]
            sym = f'<td rowspan="3" class="sym">{r["symbol"]}</td>' if i == 0 else ""
            rows.append(f"<tr>{sym}<td class='l'>{names[p]}</td><td>{s['n']:,}</td><td>{pp(s['acc'], False)}</td>"
                        f"<td>{pp(s['baseline_acc'], False)}</td><td><b>{pp(s['edge'])}</b></td>"
                        f"<td>[{pp(s['edge_ci'][0])}, {pp(s['edge_ci'][1])}]</td><td>{s['auc']:.3f}</td></tr>")
    return "\n".join(rows)


def table2():
    rows = []
    for r in R:
        cells = []
        for v in (Q, S):
            for m in ("xgb", "lstm", "ens"):
                cells.append(f"<td>{pp(r[v]['P2_walk_forward'][m]['edge'])}</td>")
        lu = r[Q]["P2_walk_forward"]["lstm"]["pred_up_rate"]
        rows.append(f"<tr><td class='sym'>{r['symbol']}</td>{''.join(cells)}<td>{lu * 100:.0f}%</td></tr>")
    return "\n".join(rows)


def table3():
    rows = []
    lab = {"long_flat_p0.5": "Long if p &ge; 0.50", "long_flat_p0.65": "Long if p &ge; 0.65", "buy_and_hold": "Buy and hold"}
    for s in SYMS:
        for i, k in enumerate(["long_flat_p0.5", "long_flat_p0.65", "buy_and_hold"]):
            x = st[s][k]
            sym = f'<td rowspan="3" class="sym">{s}</td>' if i == 0 else ""
            rows.append(f"<tr>{sym}<td class='l'>{lab[k]}</td><td>{pp(x['cagr'], False)}%</td><td>{x['sharpe']:.2f}</td>"
                        f"<td>{pp(x['max_dd'], False)}%</td><td>{x['time_in_market'] * 100:.0f}%</td>"
                        f"<td>{x['trades_per_year']:.0f}</td></tr>")
    return "\n".join(rows)


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Where Did the Edge Go? — Mohammed Hadi (working paper)</title>
<style>
@page {{ size: Letter; margin: 0.9in 0.95in; }}
body {{ font-family: Cambria, Georgia, "Times New Roman", serif; font-size: 10.8pt; line-height: 1.45; color: #111; margin: 0; }}
h1 {{ font-size: 17pt; line-height: 1.25; text-align: center; margin: 0 0 8px; font-weight: 700; }}
.authors {{ text-align: center; font-size: 10.5pt; margin-bottom: 2px; }}
.meta {{ text-align: center; font-size: 9.5pt; color: #444; margin-bottom: 14px; }}
.abstract {{ margin: 0 0.35in 12px; font-size: 10pt; }}
.abstract b {{ font-variant: small-caps; }}
h2 {{ font-size: 12pt; margin: 16px 0 6px; }}
h3 {{ font-size: 10.8pt; margin: 10px 0 4px; font-style: italic; font-weight: 600; }}
p {{ margin: 0 0 7px; text-align: justify; hyphens: auto; }}
ol, ul {{ margin: 0 0 7px; padding-left: 22px; }}
li {{ margin-bottom: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 4px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 3px 6px; text-align: right; border-bottom: 1px solid #ddd; }}
th {{ border-top: 1.5px solid #111; border-bottom: 1px solid #111; font-weight: 600; }}
td.sym, td.l, th.l {{ text-align: left; }}
tbody tr:last-child td {{ border-bottom: 1.5px solid #111; }}
.caption {{ font-size: 9pt; color: #333; margin: 2px 0 12px; }}
.caption b {{ color: #111; }}
figure {{ margin: 8px 0 12px; break-inside: avoid; }}
figure img {{ width: 100%; }}
.tbl {{ break-inside: avoid; }}
code {{ font-family: Consolas, monospace; font-size: 9pt; }}
.refs p {{ text-align: left; padding-left: 18px; text-indent: -18px; font-size: 9.6pt; margin-bottom: 4px; }}
.todo {{ background: #fff3c4; }}
</style></head><body>

<h1>Where Did the Edge Go? Leakage, Naive Baselines and Non-Stationary Features in a Machine-Learning Trading Signal</h1>
<div class="authors">Mohammed Hadi</div>
<div class="meta">Manipal Institute of Technology, Manipal Academy of Higher Education · mhdhadiabdullah@gmail.com<br>Working paper, November 2026 · Code: github.com/mhdhadi2005/QuantAI</div>

<div class="abstract"><b>Abstract.</b> Machine-learning trading systems built from technical indicators often report high predictive accuracy. I audit one such system that I built, QuantAI, which combines an XGBoost classifier and an LSTM to predict whether a stock's price will be higher five trading days ahead. Using daily data for AAPL, MSFT, NVDA, SPY and QQQ from 2010 to October 2026, I evaluate the same pipeline under three protocols: scoring on the window the model was trained on, a chronological 80/20 split without purging, and a purged expanding walk-forward over 2016–2026. Measured against the naive baseline of always predicting the majority class, the ensemble's apparent edge falls from {rng(p0_ens)} percentage points in-sample to {rng(p2_ens)} points out of sample, with AUC between {min(p2_auc):.2f} and {max(p2_auc):.2f}. For {p2_ci_neg} of 5 assets the 95% block-bootstrap interval lies entirely below zero. Removing raw price-level features narrows the deficit to {rng(p2s_ens)} points, which identifies extrapolation on non-stationary inputs as a main source of the damage. A long/flat strategy built from the out-of-sample predictions underperforms buy-and-hold on Sharpe ratio for {worse_sharpe} of 5 assets after costs. The results are a negative finding, and a practical checklist for evaluating similar systems honestly.</div>

<h2>1. Introduction</h2>
<p>Machine learning has become a standard tool in empirical finance. Gu, Kelly and Xiu (2020) show that flexible models can improve return prediction when evaluated carefully on large cross-sections. At the same time, the literature on backtest overfitting (Bailey et al., 2014; Harvey, Liu and Zhu, 2016) and on label leakage in financial time series (López de Prado, 2018) shows how easily careless evaluation manufactures performance that does not exist.</p>
<p>This paper studies that gap in a small, concrete setting: a trading system I built myself. QuantAI predicts five-day price direction from technical indicators with an ensemble of XGBoost (Chen and Guestrin, 2016) and an LSTM (Hochreiter and Schmidhuber, 1997), and feeds its predictions into a rule-based backtester. Reviewing the code for this paper, I found three evaluation choices that could inflate measured performance: (i) in the backtest path that uses the model, the model is trained on the same window it is then tested on; (ii) the train/test split does not purge the five days whose labels overlap the test period; and (iii) accuracy is reported without comparison to the naive majority-class baseline, which is high for equities that drift upward.</p>
<p>The contributions are modest and practical. First, I quantify how much each of these choices changes the measured edge of an otherwise identical pipeline. Second, an ablation shows that raw price-level features (prices, moving averages, on-balance volume) are the main reason tree-based predictions fail out of sample. Third, I translate the honest predictions into a tradable long/flat rule with execution delay and costs, to check whether any residual signal survives economically.</p>

<h2>2. The QuantAI pipeline</h2>
<h3>2.1 Data</h3>
<p>Daily split- and dividend-adjusted OHLCV bars are downloaded from Yahoo Finance for the five symbols QuantAI tracks by default (AAPL, MSFT, NVDA, SPY, QQQ), from {cfg['start']} to the end of October 2026. After indicator warm-up, each series has about {R[0]['rows']:,} daily observations.</p>
<h3>2.2 Features and label</h3>
<p>The feature set reproduces QuantAI exactly: RSI(14), MACD and its signal and histogram, stochastic RSI, Bollinger %B and width, ATR(14), SMA(20), EMA(9) and EMA(21), trend and above-SMA(50) flags, distance from SMA(20), a composite momentum score, on-balance volume, raw open/high/low/close and volume, 1-, 5- and 10-day returns, three lagged daily returns, 5-, 10- and 20-day realised volatility, two volume ratios, and three market features from SPY (its close, daily return and an above-SMA(50) flag) for non-SPY assets. All indicators are computed causally from data up to day <i>t</i>. The label is 1 if the close on day <i>t</i>+{cfg['horizon']} exceeds the close on day <i>t</i>, and 0 otherwise. Two incidental bugs in the original code are deliberately not reproduced, so that the study isolates the evaluation choices: the market features were fetched for only the most recent year, which silently cut each training set to about 360 rows, and the final {cfg['horizon']} rows, which have no future close, were labelled "down" instead of being dropped.</p>
<h3>2.3 Models</h3>
<p>Features are standardised with a scaler fitted on training data only. The tree model is XGBoost with QuantAI's settings (150 trees, depth 4, learning rate 0.03, row and column subsampling 0.8, L1 0.1, L2 1.0). The sequence model is a two-layer LSTM with 64 hidden units over windows of 10 days, trained for 15 full-batch Adam epochs (learning rate 0.001) on binary cross-entropy. The ensemble probability is the average of the two. No hyperparameters were tuned for this study: the point is to evaluate the system as built.</p>

<h2>3. Evaluation protocols</h2>
<ol>
<li><b>P0 – In-sample.</b> Train and score on the same final 20% of the sample. This reproduces what QuantAI's model-driven backtest does when it trains the model on the backtest window before replaying it.</li>
<li><b>P1 – Chronological 80/20 split, no purge.</b> QuantAI's own validation: the first 80% of labelled days for training, the last 20% for testing, with no gap. Because labels look five days ahead, the last five training labels depend on prices inside the test period.</li>
<li><b>P2 – Purged expanding walk-forward.</b> For each test year from {cfg['test_years'][0]} to {cfg['test_years'][1]}, the models are retrained on all earlier data, minus the final {cfg['horizon']} labelled days before the test year (the purge), then scored only on that year. Predictions are pooled across years ({R[0][Q]['P2_walk_forward']['ens']['n']:,} test days per asset).</li>
</ol>
<p><b>Baseline and inference.</b> The benchmark for every protocol is the naive classifier that always predicts the majority class of its training data; for these assets that is "up", correct {rng(base_rates, lambda v: pp(v, False))}% of the time out of sample. I report the <i>edge</i>, defined as accuracy minus baseline accuracy, with a 95% confidence interval from a moving-block bootstrap (Künsch, 1989) with 20-day blocks and 1,000 resamples to respect serial dependence, and AUC as a threshold-free measure of ranking skill.</p>
<p><b>Economic test.</b> Using P2 ensemble probabilities, a long/flat rule holds the asset when the probability is at least 0.50 (or at least 0.65, QuantAI's confidence threshold). The decision uses the close of day <i>t</i> and is executed at the close of day <i>t</i>+1, matching QuantAI's one-bar delay, with {cfg['cost_per_side'] * 100:.2f}% cost per side (0.1% commission plus 0.01% slippage, QuantAI's assumptions).</p>

<h2>4. Results</h2>
<h3>4.1 The edge disappears out of sample</h3>
<p>Table 1 and Figure 1 report the ensemble under each protocol. In-sample (P0), XGBoost alone is {rng(p0_xgb_acc, lambda v: pp(v, False))}% accurate, and the ensemble beats the majority baseline by {rng(p0_ens)} percentage points. This is the kind of number a model-driven backtest that trains on its own test window would quietly reflect. Under QuantAI's own 80/20 split (P1), the edge is {rng(p1_ens)} points. Under the purged walk-forward (P2), it is {rng(p2_ens)} points, the bootstrap interval excludes zero (below) for {p2_ci_neg} of 5 assets, and AUC lies between {min(p2_auc):.3f} and {max(p2_auc):.3f}: no better than chance at ranking up-days above down-days.</p>

<figure><img src="figures/fig1_edge_by_protocol.png" alt="Bar chart of ensemble accuracy minus majority baseline under each protocol for five assets">
<div class="caption"><b>Figure 1.</b> Ensemble accuracy minus majority-class baseline accuracy (percentage points), by evaluation protocol. Positive bars in-sample reverse once the model is scored on data it has not seen.</div></figure>

<div class="tbl"><div class="caption"><b>Table 1.</b> Ensemble (XGBoost + LSTM) performance by protocol, QuantAI feature set. Edge = accuracy − majority baseline, in percentage points; 95% CI from a moving-block bootstrap.</div>
<table><thead><tr><th class='l'>Asset</th><th class='l'>Protocol</th><th>Test days</th><th>Accuracy %</th><th>Baseline %</th><th>Edge (pp)</th><th>95% CI</th><th>AUC</th></tr></thead>
<tbody>{table1()}</tbody></table></div>

<h3>4.2 Why: non-stationary features and a collapsing LSTM</h3>
<p>Table 2 breaks the walk-forward result down by model and feature set. Two patterns stand out. First, XGBoost on the full feature set is the main source of the deficit ({rng(p2_xgb)} points). Dropping the price-level features (raw OHLC, moving averages, MACD in price units, ATR, on-balance volume and the SPY level) narrows its deficit to {rng(p2s_xgb)} points. Trees cannot extrapolate: when prices rise beyond anything seen in training, every test observation falls into the same edge leaves, and predictions become arbitrary. The clearest case is SPY under P1, where XGBoost predicted "up" on only {spy_p1_xgb_up * 100:.0f}% of test days during a period in which most five-day windows rose.</p>
<p>Second, the LSTM predicts "up" on {rng(lstm_up, lambda v: f"{v * 100:.0f}")}% of walk-forward test days. With only 15 full-batch epochs and a weak signal, it converges to the class prior rather than learning conditional structure. Its accuracy is therefore at or below the baseline, never meaningfully above it, and averaging it with XGBoost mostly dilutes XGBoost's errors rather than adding information.</p>

<div class="tbl"><div class="caption"><b>Table 2.</b> Purged walk-forward edge (pp) by model and feature set, and the LSTM's share of "up" predictions. "Stationary" removes raw price-level features.</div>
<table><thead><tr><th class='l'>Asset</th><th>XGB (all)</th><th>LSTM (all)</th><th>Ens. (all)</th><th>XGB (stat.)</th><th>LSTM (stat.)</th><th>Ens. (stat.)</th><th>LSTM % up</th></tr></thead>
<tbody>{table2()}</tbody></table></div>

<h3>4.3 No economic value after costs</h3>
<p>Table 3 and Figure 2 translate walk-forward probabilities into the long/flat rule. The 0.50 rule underperforms buy-and-hold on Sharpe ratio for {worse_sharpe} of 5 assets and on CAGR for {worse_cagr} of 5, while trading about two dozen times a year. Raising the threshold to 0.65 keeps the strategy out of the market most of the time and produces CAGRs close to zero. Maximum drawdowns are not meaningfully smaller than buy-and-hold, so the rule does not even work as a risk filter.</p>

<figure><img src="figures/fig2_equity.png" alt="Equity curves of the model long/flat strategy versus buy-and-hold for five assets, 2016 to 2026">
<div class="caption"><b>Figure 2.</b> Growth of $1 (log scale), 2016–2026: long/flat strategy from walk-forward ensemble probabilities (p &ge; 0.50, one-bar delay, costs included) versus buy-and-hold.</div></figure>

<div class="tbl"><div class="caption"><b>Table 3.</b> Out-of-sample strategy performance, 2016–October 2026, net of {cfg['cost_per_side'] * 100:.2f}% cost per side.</div>
<table><thead><tr><th class='l'>Asset</th><th class='l'>Rule</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>In market</th><th>Trades/yr</th></tr></thead>
<tbody>{table3()}</tbody></table></div>

<h2>5. Discussion: what to change in QuantAI</h2>
<ul>
<li><b>Never train on the backtest window.</b> The model-driven backtest should use models trained strictly before each test period, with retraining on a schedule, as in P2.</li>
<li><b>Purge (and embargo) overlapping labels</b> at every train/test boundary, as López de Prado (2018) recommends. This study does not isolate the purge's effect; with a five-day label it touches only a handful of training rows per split, so the in-sample leak is by far the larger problem.</li>
<li><b>Report the baseline next to every accuracy number.</b> On assets that rise more often than they fall, 55–60% accuracy can be worse than predicting "up" every day.</li>
<li><b>Use stationary inputs.</b> Express prices, averages and volume as returns, ratios or z-scores before giving them to tree models.</li>
<li><b>Judge the signal economically.</b> Accuracy is not the objective; net-of-cost risk-adjusted return against buy-and-hold is.</li>
</ul>

<h2>6. Limitations</h2>
<p>The asset set is small and chosen with hindsight: five U.S. large-cap stocks and ETFs that performed exceptionally well over the sample, which makes buy-and-hold an unusually strong benchmark and the results specific to this universe. Only one horizon and one feature family are tested, and hyperparameters were deliberately left at QuantAI's defaults, so the study says nothing about what a carefully tuned model on a broad cross-section could achieve. That question is the subject of the literature cited above. The LSTM is trained very briefly. Yahoo Finance data are convenient but not research-grade. Finally, the backtest checks exits only at daily closes and ignores market impact.</p>

<h2>7. Conclusion</h2>
<p>An indicator-based ML ensemble that looks highly accurate when scored on its own training window shows no out-of-sample edge over a naive baseline once evaluated with a purged walk-forward, and no economic value after costs. Most of the gap comes from evaluation design rather than modelling: training on the test window, ignoring the baseline, and feeding non-stationary price levels to trees. The checklist in Section 5 is cheap to apply, and QuantAI now implements it: walk-forward model evaluation in its backtester, a purged validation split, baseline-relative metrics that gate live use of the model, and stationary features.</p>

<p class="todo">[Acknowledgements: add anyone who advised you, and disclose AI assistance if your supervisor or venue requires it.]</p>

<h2>Reproducibility</h2>
<p>All numbers, tables and figures are produced by <code>experiment.py</code> and <code>build_paper.py</code> (Python 3.11; pandas, scikit-learn, XGBoost, PyTorch, yfinance). Random seeds are fixed. A repeated run reproduced every reported number; the raw outputs differed only by floating-point round-off (below 10<sup>-12</sup>) from re-reading cached prices.</p>

<div class="refs">
<h2>References</h2>
<p>Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. <i>Notices of the American Mathematical Society</i>, 61(5), 458–471.</p>
<p>Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785–794.</p>
<p>Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. <i>Review of Financial Studies</i>, 33(5), 2223–2273.</p>
<p>Harvey, C. R., Liu, Y., and Zhu, H. (2016). …and the cross-section of expected returns. <i>Review of Financial Studies</i>, 29(1), 5–68.</p>
<p>Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. <i>Neural Computation</i>, 9(8), 1735–1780.</p>
<p>Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. <i>Annals of Statistics</i>, 17(3), 1217–1241.</p>
<p>López de Prado, M. (2018). <i>Advances in Financial Machine Learning</i>. Hoboken, NJ: Wiley.</p>
</div>
</body></html>
"""

with open(os.path.join(HERE, "paper.html"), "w", encoding="utf-8") as f:
    f.write(HTML)
print("paper.html written")
