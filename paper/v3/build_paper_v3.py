"""Builds paper_v3.html from ../v2/results_v2.json and regularization_results.json, so every number in the
text comes from an experiment. Version 3 = version 2 plus Section 5.5 (is it just overfitting?)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "v2", "results_v2.json")))
A, P, C = D["aggregate"], D["per_stock"], D["config"]
U = D["universe"]
R = json.load(open(os.path.join(HERE, "regularization_results.json")))["summary"]
RX, RXS, RL, RL3 = R["xgb_d4_v2default"], R["xgb_stumps"], R["logit_C0.1"], R["logit_C0.001"]
best_edge = max(R.values(), key=lambda s: s["wf_edge"]["mean"])
any_pos = any(s["wf_edge"]["lo"] > 0 for s in R.values())

# Optional acknowledgements: put the text in acknowledgements.txt next to this file; omitted if absent
_ack_path = os.path.join(HERE, "acknowledgements.txt")
_ack_text = open(_ack_path, encoding="utf-8").read().strip() if os.path.exists(_ack_path) else ""
ACK = f"<h2>Acknowledgements</h2>\n<p>{_ack_text}</p>" if _ack_text else ""


def pp(x, sign=True):
    v = round(x * 100, 1) + 0.0
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def ci(e):
    return f"{pp(e['mean'])} [{pp(e['lo'])}, {pp(e['hi'])}]"


def auc(e):
    return f"{e['mean']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"


iso_x, iso_l = A["isolation"]["xgb"], A["isolation"]["logit"]
wf = A["walk_forward"]
fq, ec = A["fixed_quantai"], A["economics"]
purge_diff = np.mean([s["isolation"]["xgb"]["P1a_split_no_purge"]["edge"] - s["isolation"]["xgb"]["P1b_split_purged"]["edge"] for s in P])
sectors = {}
for u in U:
    sectors[u["sector"]] = sectors.get(u["sector"], 0) + 1
sector_text = ", ".join(f"{k} {v}" for k, v in sorted(sectors.items(), key=lambda kv: -kv[1]))
n_sig_expected = 0.05 * C["n_stocks"]


def row(label, e, a):
    return f"<tr><td class='l'>{label}</td><td>{ci(e)}</td><td>{auc(a)}</td></tr>"


def table1():
    rows = []
    for name, src in (("XGBoost", iso_x), ("Logistic regression", iso_l)):
        rows.append(f"<tr class='grp'><td class='l' colspan='3'>{name}</td></tr>")
        rows.append(row("Trained and scored on the same window (in-sample)", src["P0_in_sample"]["edge"], src["P0_in_sample"]["auc"]))
        rows.append(row("80/20 split, no purge", src["P1a_split_no_purge"]["edge"], src["P1a_split_no_purge"]["auc"]))
        rows.append(row("80/20 split, purged", src["P1b_split_purged"]["edge"], src["P1b_split_purged"]["auc"]))
        k = "xgb" if name == "XGBoost" else "logit"
        rows.append(row("Walk-forward, original features (with price levels)", wf["v1_raw"][k]["edge"], wf["v1_raw"][k]["auc"]))
        rows.append(row("Walk-forward, stationary features", wf["v2_raw"][k]["edge"], wf["v2_raw"][k]["auc"]))
    return "\n".join(rows)


def table2():
    names = [("logit", "Logistic regression"), ("xgb", "XGBoost"), ("lstm_app", "LSTM, original training"),
             ("lstm", "LSTM, proper training"), ("ens", "XGBoost + proper LSTM")]
    out = []
    for k, label in names:
        s = wf["v2_raw"][k]
        out.append(f"<tr><td class='l'>{label}</td><td>{ci(s['edge'])}</td><td>{auc(s['auc'])}</td>"
                   f"<td>{s['share_positive_edge'] * 100:.0f}%</td><td>{s['share_pt_p_below_05'] * 100:.0f}%</td>"
                   f"<td>{s['up_rate'] * 100:.0f}%</td></tr>")
    return "\n".join(out)


def table3():
    out = []
    for c, label in (("v2_raw", "Raw: price up after 5 days"), ("v2_excess", "Relative: beats SPY over 5 days")):
        for k, m in (("xgb", "XGBoost"), ("logit", "Logistic")):
            s = wf[c][k]
            out.append(f"<tr><td class='l'>{label}</td><td class='l'>{m}</td><td>{ci(s['edge'])}</td><td>{auc(s['auc'])}</td>"
                       f"<td>{s['share_positive_edge'] * 100:.0f}%</td></tr>")
    return "\n".join(out)


def table5():
    names = [("xgb_d4_v2default", "XGBoost, depth 4 (as in Table 2)"), ("xgb_d2", "XGBoost, depth 2"),
             ("xgb_stumps", "XGBoost, stumps (depth 1)"), ("xgb_stumps_small4", "XGBoost, stumps, 4 features"),
             ("logit_C1", "Logistic, C = 1"), ("logit_C0.1", "Logistic, C = 0.1 (as in Table 2)"),
             ("logit_C0.01", "Logistic, C = 0.01"), ("logit_C0.001", "Logistic, C = 0.001"),
             ("logit_L1_C0.01", "Logistic, L1, C = 0.01"), ("logit_small4", "Logistic, 4 features")]
    out = []
    for k, label in names:
        s = R[k]
        out.append(f"<tr><td class='l'>{label}</td><td>{s['train_acc']['mean'] * 100:.1f}</td>"
                   f"<td>{s['wf_acc']['mean'] * 100:.1f}</td><td>{pp(s['gap_train_minus_wf']['mean'], False)}</td>"
                   f"<td>{ci(s['wf_edge'])}</td><td>{s['wf_auc']['mean']:.3f}</td>"
                   f"<td>{s['up_rate']['mean'] * 100:.0f}%</td></tr>")
    return "\n".join(out)


def appendix():
    cells = [f"{u['symbol']} <span class='sec'>({u['sector']})</span>" for u in sorted(U, key=lambda u: u["symbol"])]
    rows = []
    for i in range(0, len(cells), 3):
        rows.append("<tr>" + "".join(f"<td class='l'>{c}</td>" for c in cells[i:i + 3]) + "</tr>")
    return "\n".join(rows)


on_share = fq["windows_gated_on"] / fq["windows"]

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Where Did the Edge Go? (v3) — Mohammed Hadi</title>
<style>
@page {{ size: Letter; margin: 0.9in 0.95in; }}
body {{ font-family: Cambria, Georgia, "Times New Roman", serif; font-size: 10.8pt; line-height: 1.45; color: #111; margin: 0; }}
h1 {{ font-size: 17pt; line-height: 1.25; text-align: center; margin: 0 0 8px; }}
.authors {{ text-align: center; font-size: 10.5pt; margin-bottom: 2px; }}
.meta {{ text-align: center; font-size: 9.5pt; color: #444; margin-bottom: 14px; }}
.abstract {{ margin: 0 0.35in 12px; font-size: 10pt; }}
.abstract b {{ font-variant: small-caps; }}
h2 {{ font-size: 12pt; margin: 16px 0 6px; break-after: avoid; }}
h3 {{ break-after: avoid; }}
h3 {{ font-size: 10.8pt; margin: 10px 0 4px; font-style: italic; font-weight: 600; }}
p {{ margin: 0 0 7px; text-align: justify; hyphens: auto; }}
ol, ul {{ margin: 0 0 7px; padding-left: 22px; }}
li {{ margin-bottom: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 4px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 3px 6px; text-align: right; border-bottom: 1px solid #ddd; }}
th {{ border-top: 1.5px solid #111; border-bottom: 1px solid #111; font-weight: 600; }}
td.l, th.l {{ text-align: left; }}
tr.grp td {{ font-weight: 600; border-bottom: 1px solid #bbb; padding-top: 6px; }}
tbody tr:last-child td {{ border-bottom: 1.5px solid #111; }}
.caption {{ font-size: 9pt; color: #333; margin: 2px 0 12px; }}
.caption b {{ color: #111; }}
figure {{ margin: 8px 0 12px; break-inside: avoid; }}
figure img {{ width: 100%; }}
figure.half img {{ width: 70%; display: block; margin: 0 auto; }}
.tbl {{ break-inside: avoid; }}
code {{ font-family: Consolas, monospace; font-size: 9pt; }}
.sec {{ color: #666; font-size: 8.5pt; }}
.refs p {{ text-align: left; padding-left: 18px; text-indent: -18px; font-size: 9.6pt; margin-bottom: 4px; }}
.todo {{ background: #fff3c4; }}
</style></head><body>

<h1>Where Did the Edge Go? Leakage, Naive Baselines and Non-Stationary Features in a Machine-Learning Trading Signal</h1>
<div class="authors">Mohammed Hadi</div>
<div class="meta">Manipal Institute of Technology, Manipal Academy of Higher Education · mhdhadiabdullah@gmail.com<br>Working paper, version 3, November 2026 · Code: github.com/mhdhadi2005/QuantAI (folder <code>paper/</code>)</div>

<div class="abstract"><b>Abstract.</b> I audit QuantAI, an indicator-based trading system I built that combines XGBoost and an LSTM to predict whether a stock will be higher five trading days ahead. Using {C['n_stocks']} stocks drawn at random from the S&amp;P 500 and daily data from 2010 to October 2026, I vary one evaluation choice at a time. Scoring a model on the window it was trained on produces an apparent edge over the naive majority-class baseline of {ci(iso_x['P0_in_sample']['edge'])} percentage points (mean across stocks, 95% interval); on a held-out chronological split the same model is at {ci(iso_x['P1b_split_purged']['edge'])} points. Purging overlapping labels changes the result by {abs(purge_diff) * 100:.1f} points, and replacing price-level features with stationary ones improves the walk-forward edge from {pp(wf['v1_raw']['xgb']['edge']['mean'])} to {pp(wf['v2_raw']['xgb']['edge']['mean'])} points. Under a purged walk-forward over 2016–2026, no model beats the baseline on average, including a logistic regression and a properly regularised LSTM (AUC {wf['v2_raw']['lstm']['auc']['mean']:.3f}); a market-relative label gives {ci(wf['v2_excess']['xgb']['edge'])} points. The fixed version of QuantAI, which uses the model only when its validation edge exceeds its margin of error, switches the model on in {fq['windows_gated_on']} of {fq['windows']} stock-years, and those choices do not outperform. The original XGBoost is heavily overfitted (in-sample accuracy {RX['train_acc']['mean'] * 100:.0f}% against {RX['wf_acc']['mean'] * 100:.0f}% walk-forward), but restricting it in-sample only removes harm: across ten regularization settings the walk-forward edge rises to at best {pp(best_edge['wf_edge']['mean'])} points and never exceeds zero. Daily technical indicators carry no exploitable directional signal in this sample; the most consequential error is evaluating on training data, and the most useful fix is a gate that keeps a signal-free model out of trading.</div>

<h2>1. Introduction</h2>
<p>Machine learning is now routine in empirical finance. Gu, Kelly and Xiu (2020) show that flexible models can improve return prediction when evaluated carefully on a large cross-section of stocks. The literature on backtest overfitting (Bailey et al., 2014; Harvey, Liu and Zhu, 2016) and on leakage in financial labels (López de Prado, 2018) shows how easily careless evaluation manufactures performance that is not there.</p>
<p>This paper studies that gap in a concrete system that I built. QuantAI predicts five-day price direction for individual stocks from technical indicators, using an ensemble of XGBoost (Chen and Guestrin, 2016) and an LSTM (Hochreiter and Schmidhuber, 1997). A first version of this paper, on five hand-picked large-cap assets, found that the system's apparent accuracy disappeared under honest evaluation. This version answers the objections that study left open:</p>
<ol>
<li><b>Which evaluation choice matters?</b> I change one at a time: training on the test window, purging overlapping labels, price-level features, and the label definition.</li>
<li><b>Was the LSTM just badly trained?</b> I add a properly regularised LSTM with mini-batches and early stopping, and a logistic regression as a simple benchmark.</li>
<li><b>Was the sample cherry-picked?</b> I use {C['n_stocks']} stocks drawn at random from the S&amp;P 500 rather than five winners.</li>
<li><b>Does the fix work?</b> I evaluate the corrected QuantAI, including its new rule for when to trust the model, with the same protocol.</li>
<li><b>Is it just overfitting?</b> New in version 3, in response to reader feedback on version 2: I restrict the models in-sample across ten regularization settings and test whether out-of-sample skill appears.</li>
</ol>

<h2>2. Data</h2>
<p>I took the 503 constituents of the S&amp;P 500 listed on 4 November 2026, shuffled them with a fixed seed ({C['seed']}), and kept the first {C['n_stocks']} with daily data from early 2010, so that every walk-forward year has at least five years of training history. {len(D['skipped'])} stocks were skipped for lack of history. The sample spans 11 sectors ({sector_text}); the list is in the appendix. Split- and dividend-adjusted daily bars come from Yahoo Finance for {C['start'][:4]} to October 2026, and SPY supplies the market features and the market-relative label.</p>

<h2>3. Pipeline</h2>
<h3>3.1 Features</h3>
<p>The <i>original</i> feature set reproduces QuantAI before its fix: RSI, MACD and its signal and histogram, stochastic RSI, Bollinger %B and width, ATR, SMA(20), EMA(9) and EMA(21), trend flags, distance from SMA(20), a momentum score, on-balance volume, raw open/high/low/close and volume, recent returns and their lags, realised volatility, volume ratios, and SPY's level, return and trend. The <i>stationary</i> set is the fixed version: price-unit quantities are divided by price (ATR, MACD, the EMA spread), on-balance volume becomes a 20-day flow ratio, and all raw price and volume levels are removed. All features use data up to day <i>t</i> only.</p>
<h3>3.2 Labels</h3>
<p>The <i>raw</i> label is 1 if the close on day <i>t</i>+{C['horizon']} exceeds the close on day <i>t</i>. The <i>relative</i> label is 1 if the stock's five-day return exceeds SPY's. The raw label inherits the market's upward drift, so the naive baseline is well above 50%; the relative label removes that drift.</p>
<h3>3.3 Models</h3>
<p>All models see standardised features with the scaler fitted on training data only. <b>Logistic regression</b> uses L2 regularisation (C = 0.1). <b>XGBoost</b> uses QuantAI's settings (150 trees, depth 4, learning rate 0.03, subsampling 0.8). The <b>original LSTM</b> is QuantAI's: two layers of 64 units over 10-day windows, trained for 15 full-batch Adam epochs. The <b>proper LSTM</b> has one layer of 32 units with dropout 0.2 and weight decay, trained in mini-batches of 128 for up to 40 epochs, with early stopping (patience 4) on the last 15% of each training window, purged by {C['horizon']} days, and the best weights restored. The <b>ensemble</b> averages XGBoost and the proper LSTM. No hyperparameters were tuned on test data.</p>

<h2>4. Evaluation design</h2>
<p><b>Isolating each choice.</b> On the final 20% of each stock's sample (about the last three years), I score the original feature set three ways: trained on that same window (in-sample), trained on the preceding 80% with no gap, and trained on the preceding 80% with the last {C['horizon']} labelled days removed (purged). The first comparison isolates training on the test window; the second isolates purging.</p>
<p><b>Walk-forward.</b> For each year from {C['test_years'][0]} to {C['test_years'][1]}, every model is retrained on all earlier data minus a {C['horizon']}-day purge and scored only on that year. Predictions are pooled across years for each stock.</p>
<p><b>Metrics and inference.</b> The <i>edge</i> is accuracy minus the accuracy of always predicting the majority class of the training data. I also report AUC, the share of stocks with a positive edge, and the share with a significant Pesaran–Timmermann (1992) test of directional skill at the 5% level, computed on every fifth day so that labels do not overlap (about {n_sig_expected:.1f} of {C['n_stocks']} stocks would pass by chance). Cross-stock means carry 95% <i>t</i>-intervals that treat stocks as independent; because stocks share market moves, these intervals are somewhat too narrow.</p>
<p><b>The fixed QuantAI.</b> QuantAI now trains its ensemble on 80% of the available history, measures its edge on the remaining 20% (purged), and uses the model in trading only if that edge exceeds its 95% margin of error. I replicate this inside the walk-forward for every stock-year, then record whether the gate opened and how the model did on the following year.</p>
<p><b>Economic test.</b> A long/flat rule holds a stock when the predicted probability of an up move is at least 0.5, deciding at the close of day <i>t</i> and trading at the close of day <i>t</i>+1, with {C['cost_per_side'] * 100:.2f}% cost per side. For the fixed QuantAI the stock is simply held when the gate is closed, as the live system would then ignore the model.</p>

<h2>5. Results</h2>
<h3>5.1 Where the apparent edge comes from</h3>
<p>Table 1 and Figure 1 change one choice at a time. Training on the test window accounts for essentially all of the apparent skill: XGBoost's edge falls from {ci(iso_x['P0_in_sample']['edge'])} points in-sample, with an AUC of {iso_x['P0_in_sample']['auc']['mean']:.3f}, to {ci(iso_x['P1b_split_purged']['edge'])} on a held-out split. Logistic regression, which can memorise far less, still shows {pp(iso_l['P0_in_sample']['edge']['mean'])} points in-sample. Purging the {C['horizon']} overlapping days changes the held-out edge by only {abs(purge_diff) * 100:.1f} points on average: with a five-day label and thousands of training days the overlap is too small to matter. Version 1 of this paper listed purging as a fix without measuring it; this result shows it is not where the damage was. Under the walk-forward, removing price-level features improves XGBoost's edge from {pp(wf['v1_raw']['xgb']['edge']['mean'])} to {pp(wf['v2_raw']['xgb']['edge']['mean'])} points, consistent with tree models failing to extrapolate when prices move beyond the training range.</p>

<figure><img src="figures/fig1_isolation.png" alt="Bar chart of XGBoost edge over the majority baseline as each evaluation choice is changed">
<div class="caption"><b>Figure 1.</b> Mean XGBoost edge over the majority-class baseline across {C['n_stocks']} stocks, with 95% intervals, as each evaluation choice is changed. The first three bars use the final 20% of each sample with the original features; the last two are the 2016–2026 walk-forward.</div></figure>

<div class="tbl"><div class="caption"><b>Table 1.</b> Mean edge (percentage points) and AUC across {C['n_stocks']} stocks, raw label, with 95% intervals.</div>
<table><thead><tr><th class='l'>Evaluation</th><th>Edge (pp)</th><th>AUC</th></tr></thead>
<tbody>{table1()}</tbody></table></div>

<h3>5.2 No model beats the baseline</h3>
<p>Table 2 and Figure 2 compare models under the walk-forward with stationary features. Every model's mean edge is negative, and few stocks show a positive edge for any model. The original LSTM predicts "up" on {wf['v2_raw']['lstm_app']['up_rate'] * 100:.0f}% of days: it has learned the class prior, which is why its edge ({pp(wf['v2_raw']['lstm_app']['edge']['mean'])}) is close to zero rather than evidence of skill. Training the LSTM properly does not change the picture: its edge is {ci(wf['v2_raw']['lstm']['edge'])} points. Its AUC of {auc(wf['v2_raw']['lstm']['auc'])} is the highest of the models and its interval excludes 0.5, but an AUC of about 0.51 is far too small to trade on, and the interval is optimistic because stocks are correlated. The Pesaran–Timmermann test is significant for {wf['v2_raw']['lstm']['share_pt_p_below_05'] * 100:.0f}% of stocks for the proper LSTM and between {min(wf['v2_raw'][k]['share_pt_p_below_05'] for k in wf['v2_raw']) * 100:.0f}% and {max(wf['v2_raw'][k]['share_pt_p_below_05'] for k in wf['v2_raw']) * 100:.0f}% across models, against 5% expected by chance. That is weak at best, given that five models were tried. The simple logistic regression ({pp(wf['v2_raw']['logit']['edge']['mean'])}) does no worse than XGBoost ({pp(wf['v2_raw']['xgb']['edge']['mean'])}), so the extra flexibility buys nothing.</p>

<figure><img src="figures/fig2_models.png" alt="Box plots of walk-forward edge across 50 stocks for five models">
<div class="caption"><b>Figure 2.</b> Walk-forward edge for each of the {C['n_stocks']} stocks (dots) by model, stationary features, raw label, 2016–2026.</div></figure>

<div class="tbl"><div class="caption"><b>Table 2.</b> Walk-forward performance by model (stationary features, raw label). "PT &lt; 0.05": share of stocks with significant directional skill (5% expected by chance). "Says up": share of days the model predicts an up move.</div>
<table><thead><tr><th class='l'>Model</th><th>Edge (pp)</th><th>AUC</th><th>Edge &gt; 0</th><th>PT &lt; 0.05</th><th>Says up</th></tr></thead>
<tbody>{table2()}</tbody></table></div>

<h3>5.3 A fairer label does not reveal hidden skill</h3>
<p>One might suspect the raw label hides skill behind the market's drift. With the relative label, the baseline falls to about 50% and the edge is {ci(wf['v2_excess']['xgb']['edge'])} points for XGBoost and {ci(wf['v2_excess']['logit']['edge'])} for logistic regression. That is zero within the interval; {wf['v2_excess']['xgb']['share_positive_edge'] * 100:.0f}% of stocks show a positive edge, close to the half expected by chance. AUC is {wf['v2_excess']['xgb']['auc']['mean']:.3f} (Table 3, Figure 3).</p>

<figure class="half"><img src="figures/fig3_label_auc.png" alt="AUC per stock for raw and market-relative labels">
<div class="caption"><b>Figure 3.</b> XGBoost walk-forward AUC per stock (dots) and mean (line), for the raw and the market-relative label.</div></figure>

<div class="tbl"><div class="caption"><b>Table 3.</b> Walk-forward results by label definition (stationary features).</div>
<table><thead><tr><th class='l'>Label</th><th class='l'>Model</th><th>Edge (pp)</th><th>AUC</th><th>Edge &gt; 0</th></tr></thead>
<tbody>{table3()}</tbody></table></div>

<h3>5.4 The fixed QuantAI</h3>
<p>The corrected system's gate opened in {fq['windows_gated_on']} of {fq['windows']} stock-years ({on_share * 100:.1f}%), for {fq['stocks_ever_on']} different stocks. When it opened, the model's edge over the following year was {ci(fq['edge_when_on'])} points, no better than in the years it stayed closed ({ci(fq['edge_when_off'])}). The rare passes are therefore false positives, as expected when nearly every model has no skill: the gate's value is in how seldom it opens. Economically, the long/flat rule built on the ensemble lowers the Sharpe ratio relative to buy-and-hold by {ec['ensemble_v2']['sharpe_minus_bh']['mean']:+.3f} [{ec['ensemble_v2']['sharpe_minus_bh']['lo']:+.3f}, {ec['ensemble_v2']['sharpe_minus_bh']['hi']:+.3f}] and beats it for {ec['ensemble_v2']['share_beating_bh'] * 100:.0f}% of stocks. The gated system stays in the market {ec['fixed_quantai_gated']['time_in_market'] * 100:.0f}% of the time and matches buy-and-hold almost exactly (Sharpe difference {ec['fixed_quantai_gated']['sharpe_minus_bh']['mean']:+.3f} [{ec['fixed_quantai_gated']['sharpe_minus_bh']['lo']:+.3f}, {ec['fixed_quantai_gated']['sharpe_minus_bh']['hi']:+.3f}]). The fix does not create an edge; it stops a signal-free model from costing money.</p>

<div class="tbl"><div class="caption"><b>Table 4.</b> The fixed QuantAI and the economic test, 2016–October 2026, net of {C['cost_per_side'] * 100:.2f}% cost per side.</div>
<table><thead><tr><th class='l'>Measure</th><th>Value</th></tr></thead><tbody>
<tr><td class='l'>Stock-years where the gate opened</td><td>{fq['windows_gated_on']} of {fq['windows']} ({on_share * 100:.1f}%)</td></tr>
<tr><td class='l'>Edge in the following year when the gate opened (pp)</td><td>{ci(fq['edge_when_on'])}</td></tr>
<tr><td class='l'>Edge in the following year when it stayed closed (pp)</td><td>{ci(fq['edge_when_off'])}</td></tr>
<tr><td class='l'>Ensemble long/flat: Sharpe minus buy-and-hold</td><td>{ec['ensemble_v2']['sharpe_minus_bh']['mean']:+.3f} [{ec['ensemble_v2']['sharpe_minus_bh']['lo']:+.3f}, {ec['ensemble_v2']['sharpe_minus_bh']['hi']:+.3f}]</td></tr>
<tr><td class='l'>Ensemble long/flat: stocks beating buy-and-hold</td><td>{ec['ensemble_v2']['share_beating_bh'] * 100:.0f}%</td></tr>
<tr><td class='l'>Fixed QuantAI (gated): Sharpe minus buy-and-hold</td><td>{ec['fixed_quantai_gated']['sharpe_minus_bh']['mean']:+.3f} [{ec['fixed_quantai_gated']['sharpe_minus_bh']['lo']:+.3f}, {ec['fixed_quantai_gated']['sharpe_minus_bh']['hi']:+.3f}]</td></tr>
</tbody></table></div>

<h3>5.5 Is it just overfitting?</h3>
<p>A natural reading of Sections 5.1 and 5.2 is that the models overfit: they capture noise in the training data, so they fail out of sample, and a more restricted model might do better. Table 5 and Figure 4 test this directly. On the same {C['n_stocks']} stocks, stationary features, raw label and purged walk-forward, I fit XGBoost with progressively shallower trees, stronger penalties and larger minimum leaf sizes, down to single-split stumps, and logistic regression with the L2 penalty strengthened a thousandfold, an L1 (sparse) variant, and a four-feature version. I record each model's accuracy on its own training data and its walk-forward accuracy. The difference between the two measures how much the model memorises.</p>
<p>The overfitting diagnosis is correct. The XGBoost configuration used by QuantAI fits its training data at {RX['train_acc']['mean'] * 100:.1f}% accuracy but scores {RX['wf_acc']['mean'] * 100:.1f}% walk-forward, a gap of {pp(RX['gap_train_minus_wf']['mean'], False)} points. Restricting it to stumps closes the gap to {pp(RXS['gap_train_minus_wf']['mean'], False)} points. Its walk-forward edge rises in step, from {pp(RX['wf_edge']['mean'])} to {pp(RXS['wf_edge']['mean'])} points; for logistic regression it rises from {pp(RL['wf_edge']['mean'])} at C = 0.1 to {pp(RL3['wf_edge']['mean'])} at C = 0.001. Restricting the models in-sample does improve them out of sample.</p>
<p>It does not, however, produce skill. {"No" if not any_pos else "Not every"} setting has a 95% interval for the edge above zero; the best mean edge, {pp(best_edge['wf_edge']['mean'])} points, is still negative, and it is the best of ten settings chosen with hindsight. AUC stays between {min(s['wf_auc']['mean'] for s in R.values()):.3f} and {max(s['wf_auc']['mean'] for s in R.values()):.3f} throughout: at or slightly below the 0.5 of random ranking, and never meaningfully above it. The mechanism is visible in the last column of Table 5: as regularization tightens, the models predict an up move on more days, from {RX['up_rate']['mean'] * 100:.0f}% for the default XGBoost to {RXS['up_rate']['mean'] * 100:.0f}% for stumps and {RL3['up_rate']['mean'] * 100:.0f}% for the most penalised logistic regression. They converge on the naive baseline. Overfitting explains why the original model did <i>worse</i> than the baseline, but not why nothing beats it: with these features, a model that fits less loses less, and the best it can do is to stop pretending and predict the base rate.</p>

<figure><img src="figures/fig4_regularization.png" alt="Bars showing the train minus walk-forward accuracy gap shrinking with regularization, and a line showing the walk-forward edge rising towards but staying below zero">
<div class="caption"><b>Figure 4.</b> As regularization increases (left to right), the gap between in-sample and walk-forward accuracy (bars) shrinks and the walk-forward edge (line, with 95% intervals) rises towards zero but stays below it. Means across {C['n_stocks']} stocks, stationary features, raw label, 2016–2026.</div></figure>

<div class="tbl"><div class="caption"><b>Table 5.</b> Regularization sweep under the purged walk-forward (stationary features, raw label). "Train": in-sample accuracy (%, mean over retraining years). "WF": walk-forward accuracy (%). "Gap": train minus walk-forward (pp). "Says up": share of days predicted up. The baseline accuracy is {RX['wf_base']['mean'] * 100:.1f}% for every row.</div>
<table><thead><tr><th class='l'>Model</th><th>Train</th><th>WF</th><th>Gap</th><th>Edge (pp)</th><th>AUC</th><th>Says up</th></tr></thead>
<tbody>{table5()}</tbody></table></div>

<h2>6. Discussion</h2>
<p>Three lessons carry over to any similar system. First, the decisive error is scoring a model on data it was trained on; in QuantAI this happened silently inside the backtester, and it inflated the edge by about {pp(iso_x['P0_in_sample']['edge']['mean'] - iso_x['P1b_split_purged']['edge']['mean'], False)} points. Second, every accuracy figure needs its naive baseline beside it: on rising stocks, 55% accuracy is routinely worse than always predicting "up". Third, a model should earn its place in trading by beating that baseline by more than its margin of error. Because QuantAI now enforces this, it does no harm even though it has no edge. A fourth lesson comes from Section 5.5: the gap between in-sample and out-of-sample accuracy shows whether a model is overfitting, but closing that gap does not create signal. Regularization makes a model honest about what it knows, and here that turns out to be very little.</p>
<p>The negative result is about the problem, not only the models: five-day direction from daily technical indicators appears close to unpredictable for large U.S. stocks. More promising targets are those with established predictability, such as volatility (Corsi, 2009), and cross-sectional questions of which stocks will outperform others, where Gu, Kelly and Xiu (2020) find gains from machine learning. Those are the next steps for this project.</p>

<h2>7. Limitations</h2>
<p>The universe is drawn from <i>current</i> S&amp;P 500 members, so firms that were removed or failed since 2010 are absent; this survivorship bias favours buy-and-hold and is unlikely to change the directional results. Cross-stock intervals assume independence and are therefore too narrow. Only one horizon is studied, hyperparameters are not tuned beyond the regularization sweep in Section 5.5 (by design, to evaluate the system as built), exits are checked only at daily closes, and Yahoo Finance data are convenient rather than research-grade.</p>

<h2>8. Conclusion</h2>
<p>An indicator-based ML ensemble that appears {pp(iso_x['P0_in_sample']['edge']['mean'], False)} points better than a naive baseline when scored on its own training window shows no edge on {C['n_stocks']} randomly chosen stocks once evaluated honestly, whichever model, feature set or label is used. Training on the test window, not label overlap, is what manufactured the apparent skill, and regularizing the models removes their overfitting without revealing any edge. The corrected system recognises that it has no edge and keeps the model out of trading, which is the right behaviour for a model without signal.</p>

{ACK}

<h2>Reproducibility</h2>
<p>All results, tables and figures are produced by <code>paper/v2/experiment_v2.py</code>, <code>paper/v3/regularization.py</code>, <code>paper/v3/make_fig4.py</code> and <code>paper/v3/build_paper_v3.py</code> in the QuantAI repository (Python 3.11; pandas, scikit-learn, XGBoost, PyTorch, yfinance, SciPy). The constituent list used for sampling is saved alongside the code, and all random seeds are fixed.</p>

<div class="refs">
<h2>References</h2>
<p>Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. <i>Notices of the American Mathematical Society</i>, 61(5), 458–471.</p>
<p>Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785–794.</p>
<p>Corsi, F. (2009). A simple approximate long-memory model of realized volatility. <i>Journal of Financial Econometrics</i>, 7(2), 174–196.</p>
<p>Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. <i>Review of Financial Studies</i>, 33(5), 2223–2273.</p>
<p>Harvey, C. R., Liu, Y., and Zhu, H. (2016). …and the cross-section of expected returns. <i>Review of Financial Studies</i>, 29(1), 5–68.</p>
<p>Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. <i>Neural Computation</i>, 9(8), 1735–1780.</p>
<p>López de Prado, M. (2018). <i>Advances in Financial Machine Learning</i>. Hoboken, NJ: Wiley.</p>
<p>Pesaran, M. H., and Timmermann, A. (1992). A simple nonparametric test of predictive performance. <i>Journal of Business &amp; Economic Statistics</i>, 10(4), 461–465.</p>
</div>

<h2>Appendix: the {C['n_stocks']} stocks</h2>
<div class="tbl"><table><tbody>{appendix()}</tbody></table></div>
</body></html>
"""

with open(os.path.join(HERE, "paper_v3.html"), "w", encoding="utf-8") as f:
    f.write(HTML)
print("paper_v3.html written")
