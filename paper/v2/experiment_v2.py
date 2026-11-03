"""
Where Did the Edge Go? -- version 2 experiment.

Extends experiment.py (v1) in four ways:
  1. Models: adds a logistic-regression baseline and a properly trained LSTM
     (mini-batches, dropout, early stopping on a purged chronological validation
     slice), next to QuantAI's XGBoost and its original 15-epoch full-batch LSTM.
  2. Universe and label: 50 S&P 500 stocks drawn at random (fixed seed), and a
     market-relative label (beat SPY over the next 5 days) next to the raw one.
  3. Isolation: each evaluation choice is varied on its own -- training on the
     test window, purging, price-level features, the label.
  4. Fixed QuantAI: the post-fix pipeline (stationary features, purged internal
     split, gate = edge larger than its 95% margin) evaluated walk-forward.

Run:  python experiment_v2.py   (downloads data on first run; ~10-20 min on 16 cores)
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

START, END = "2010-01-01", "2026-11-01"
HORIZON = 5
TEST_YEARS = list(range(2016, 2026 + 1))
SEED = 2026
N_STOCKS = 50
COST_PER_SIDE = 0.0011  # 0.1% commission + 0.01% slippage, QuantAI's assumptions

PRICE_LEVEL = ["open", "high", "low", "close", "sma_20", "ema_9", "ema_21", "obv", "atr_14", "macd",
               "macd_signal", "macd_histogram", "index_close", "volume"]


# ------------------------------------------------------------------ data
def load(symbol):
    import yfinance as yf
    path = os.path.join(DATA, f"{symbol}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(path)
    return df


def pick_universe():
    """Shuffle the S&P 500 list with a fixed seed; keep the first N with data from 2010."""
    cons = pd.read_csv(os.path.join(HERE, "sp500_constituents_2026-11-04.csv"))
    order = np.random.default_rng(SEED).permutation(len(cons))
    chosen, skipped = [], []
    for i in order:
        sym = cons.iloc[i]["Symbol"]
        df = load(sym)
        if len(df) > 3500 and df.index[0] <= pd.Timestamp("2010-03-01"):
            chosen.append({"symbol": sym, "sector": cons.iloc[i]["GICS Sector"]})
        else:
            skipped.append(sym)
        if len(chosen) == N_STOCKS:
            break
    return chosen, skipped


# -------------------------------------------------------------- features
def indicators(df):
    """QuantAI backend/app/data/indicators.py (causal)."""
    df = df.copy()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    d = df["close"].diff()
    ag = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    r = df["rsi_14"]
    k = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min() + 1e-10)
    df["stoch_rsi_k"] = k.rolling(3).mean() * 100
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()
    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    up, lo = m + 2 * s, m - 2 * s
    df["bb_width"] = (up - lo) / m
    df["bb_pct"] = (df["close"] - lo) / (up - lo + 1e-10)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=13, adjust=False).mean()
    df["obv"] = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()
    df["trend_up"] = (df["ema_9"] > df["ema_21"]).astype(int)
    df["above_sma_50"] = (df["close"] > df["sma_50"]).astype(int)
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["momentum_score"] = ((df["rsi_14"] - 50) / 50 * 0.5
                            + (df["macd_histogram"] / (df["atr_14"] + 1e-10)).clip(-1, 1) * 0.3
                            + df["trend_up"] * 0.2)
    return df


def build(raw, spy):
    """Both feature sets (original QuantAI, fixed QuantAI) and both labels for one stock."""
    df = indicators(raw)
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    for lag in (1, 2, 3):
        df[f"return_1_lag{lag}"] = df["return_1"].shift(lag)
    for w in (5, 10, 20):
        df[f"volatility_{w}"] = df["return_1"].rolling(w).std()
    df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["volume_momentum"] = df["volume"].pct_change(5)
    # Stationary versions added by the fix (backend/app/models/predictor.py, FEATURE_VERSION 2)
    df["atr_pct"] = df["atr_14"] / df["close"]
    df["macd_pct"] = df["macd"] / df["close"]
    df["macd_signal_pct"] = df["macd_signal"] / df["close"]
    df["macd_hist_pct"] = df["macd_histogram"] / df["close"]
    df["ema_spread"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"]
    df["obv_flow_20"] = df["obv"].diff(20) / df["volume"].rolling(20).sum()
    ix = pd.DataFrame(index=spy.index)
    ix["index_close"] = spy["close"]
    ix["index_return_1"] = spy["close"].pct_change()
    ix["index_above_sma50"] = (spy["close"] > spy["close"].rolling(50).mean()).astype(int)
    df = df.join(ix, how="left")

    fut = df["close"].shift(-HORIZON)
    spy_c = spy["close"].reindex(df.index)
    spy_fut = spy_c.shift(-HORIZON)
    df["y_raw"] = np.where(fut.isna(), np.nan, (fut > df["close"]).astype(float))
    excess = fut / df["close"] - spy_fut / spy_c
    df["y_excess"] = np.where(excess.isna(), np.nan, (excess > 0).astype(float))

    v1 = ["rsi_14", "macd", "macd_signal", "macd_histogram", "stoch_rsi_k", "stoch_rsi_d", "bb_pct", "bb_width",
          "atr_14", "sma_20", "ema_9", "ema_21", "trend_up", "above_sma_50", "price_vs_sma20", "momentum_score",
          "obv", "open", "high", "low", "close", "volume", "return_1", "return_5", "return_10", "return_1_lag1",
          "return_1_lag2", "return_1_lag3", "volatility_5", "volatility_10", "volatility_20", "volume_sma_ratio",
          "volume_momentum", "index_close", "index_return_1", "index_above_sma50"]
    v2 = ["rsi_14", "stoch_rsi_k", "stoch_rsi_d", "bb_pct", "bb_width", "trend_up", "above_sma_50",
          "price_vs_sma20", "momentum_score", "atr_pct", "macd_pct", "macd_signal_pct", "macd_hist_pct",
          "ema_spread", "obv_flow_20", "return_1", "return_5", "return_10", "return_1_lag1", "return_1_lag2",
          "return_1_lag3", "volatility_5", "volatility_10", "volatility_20", "volume_sma_ratio",
          "volume_momentum", "index_return_1", "index_above_sma50"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(set(v1 + v2)))
    return df, v1, v2


# ---------------------------------------------------------------- models
def make_logit():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=2000)


def make_xgb():
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
                             random_state=42, verbosity=0, n_jobs=1)


def seqs(X, L=10):
    idx = np.arange(L - 1, len(X))
    return np.stack([X[i - L + 1:i + 1] for i in idx]), idx


def lstm_probs(Xtr, ytr, Xall, proper, seed):
    """Fit an LSTM on training rows; return P(up) for every row of Xall (rows before day 10 get 0.5).
    proper=False: QuantAI's original (2x64, 15 full-batch epochs).
    proper=True: 1x32 with dropout, mini-batches, early stopping on the last 15% of training
    (chronological, purged by HORIZON), best weights restored."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class Net(nn.Module):
        def __init__(self, d, h, layers, drop):
            super().__init__()
            self.lstm = nn.LSTM(d, h, layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(h, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(self.drop(o[:, -1, :])).squeeze(-1)

    S, idx = seqs(Xtr)
    yS = ytr[idx].astype(np.float32)
    xt, yt = torch.tensor(S, dtype=torch.float32), torch.tensor(yS)
    loss_fn = nn.BCEWithLogitsLoss()
    if not proper:
        net = Net(Xtr.shape[1], 64, 2, 0.0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(15):
            opt.zero_grad()
            loss_fn(net(xt), yt).backward()
            opt.step()
    else:
        net = Net(Xtr.shape[1], 32, 1, 0.2)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        n = len(xt)
        cut = int(n * 0.85)
        tr_i, va_i = np.arange(0, max(cut - HORIZON, 1)), np.arange(cut, n)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(seed)
        for _ in range(40):
            net.train()
            perm = tr_i[torch.randperm(len(tr_i), generator=g).numpy()]
            for b in range(0, len(perm), 128):
                j = perm[b:b + 128]
                opt.zero_grad()
                loss_fn(net(xt[j]), yt[j]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(xt[va_i]), yt[va_i]).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 4:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
    net.eval()
    Sall, idx_all = seqs(Xall)
    out = np.full(len(Xall), 0.5)
    with torch.no_grad():
        out[idx_all] = torch.sigmoid(net(torch.tensor(Sall, dtype=torch.float32))).numpy()
    return out


def fit_predict(kind, Xtr, ytr, Xte_or_all, test_mask=None, seed=0):
    """Return P(up) for the test rows. Xtr/ytr: training rows. For LSTMs pass the full matrix and a mask."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    if kind in ("logit", "xgb"):
        model = make_logit() if kind == "logit" else make_xgb()
        model.fit(sc.transform(Xtr), ytr)
        return model.predict_proba(sc.transform(Xte_or_all))[:, 1]
    p = lstm_probs(sc.transform(Xtr), ytr, sc.transform(Xte_or_all), proper=(kind == "lstm"), seed=seed)
    return p[test_mask]


# ---------------------------------------------------------------- metrics
def pt_test(y, pred):
    """Pesaran-Timmermann (1992) test of directional skill, one-sided p-value.
    Uses every HORIZON-th day so the 5-day labels do not overlap."""
    from scipy.stats import norm
    y, pred = np.asarray(y)[::HORIZON], np.asarray(pred)[::HORIZON]
    n = len(y)
    py, px, p = y.mean(), pred.mean(), (y == pred).mean()
    ps = py * px + (1 - py) * (1 - px)
    vp = ps * (1 - ps) / n
    vs = ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n + 4 * py * px * (1 - py) * (1 - px) / n ** 2
    if vp - vs <= 0:
        return 1.0  # constant predictions carry no directional information
    return float(1 - norm.cdf((p - ps) / np.sqrt(vp - vs)))


def score(y, p, base_class):
    from sklearn.metrics import roc_auc_score
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    base = float((y == base_class).mean())
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    return {"n": int(len(y)), "acc": acc, "base": base, "edge": acc - base, "auc": auc,
            "up_rate": float(pred.mean()), "pt_p": pt_test(y, pred)}


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 2 and r.std() > 0 else 0.0


# ------------------------------------------------------------- per stock
def run_stock(args):
    sym, raw, spy = args
    df, v1, v2 = build(raw, spy)
    out = {"symbol": sym, "rows": int(len(df))}
    n = len(df)
    ar = np.arange(n)
    yr = df["y_raw"].values

    # --- Isolation on the final 20% window: original features, raw label
    lab = ~np.isnan(yr)
    L = np.where(lab)[0]
    cut = L[int(len(L) * 0.8)]
    te = lab & (ar >= cut)
    X1 = df[v1].values
    iso = {}
    for kind in ("xgb", "logit"):
        row = {}
        for name, tr in (("P0_in_sample", te),
                         ("P1a_split_no_purge", lab & (ar < cut)),
                         ("P1b_split_purged", lab & (ar < cut - HORIZON))):
            p = fit_predict(kind, X1[tr], yr[tr], X1[te])
            row[name] = score(yr[te], p, int(yr[tr].mean() >= 0.5))
        iso[kind] = row
    out["isolation"] = iso

    # --- Purged walk-forward
    combos = {
        "v1_raw": (v1, "y_raw", ["xgb", "logit"]),
        "v2_raw": (v2, "y_raw", ["xgb", "logit", "lstm", "lstm_app"]),
        "v2_excess": (v2, "y_excess", ["xgb", "logit"]),
    }
    wf = {c: {m: {"y": [], "p": [], "b": []} for m in models + (["ens"] if c == "v2_raw" else [])}
          for c, (_, _, models) in combos.items()}
    app = {"y": [], "p": [], "b": [], "gate": [], "years": []}
    probs_ens = pd.Series(np.nan, index=df.index)
    probs_app = pd.Series(np.nan, index=df.index)
    for k, year in enumerate(TEST_YEARS):
        in_year = df.index.year == year
        if not in_year.any():
            continue
        first = np.argmax(in_year)
        for c, (cols, ycol, models) in combos.items():
            y = df[ycol].values
            labc = ~np.isnan(y)
            te_c = labc & in_year
            tr_c = labc & (ar < first - HORIZON)
            if te_c.sum() == 0 or tr_c.sum() < 500:
                continue
            X = df[cols].values
            base_class = int(y[tr_c].mean() >= 0.5)
            preds = {}
            for m in models:
                seed = SEED + 1000 * k + {"lstm": 1, "lstm_app": 2}.get(m, 0)  # fixed, not hash(): reproducible
                if m.startswith("lstm"):
                    preds[m] = fit_predict(m, X[tr_c], y[tr_c], X, te_c, seed=seed)
                else:
                    preds[m] = fit_predict(m, X[tr_c], y[tr_c], X[te_c])
            if c == "v2_raw":
                preds["ens"] = (preds["xgb"] + preds["lstm"]) / 2
                probs_ens[te_c] = preds["ens"]
            for m, p in preds.items():
                wf[c][m]["y"].append(y[te_c])
                wf[c][m]["p"].append(p)
                wf[c][m]["b"].append(np.full(te_c.sum(), base_class))

        # --- Fixed QuantAI: model trained on the inner 80% (purged), gate from the inner 20%
        y = yr
        labc = ~np.isnan(y)
        te_c = labc & in_year
        idx_tr = np.where(labc & (ar < first - HORIZON))[0]
        if te_c.sum() and len(idx_tr) >= 500:
            X = df[v2].values
            ic = int(len(idx_tr) * 0.8)
            itr, iva = idx_tr[:ic - HORIZON], idx_tr[ic:]
            mask_va = np.zeros(n, bool); mask_va[iva] = True
            seed = SEED + 7 * k
            pv = (fit_predict("xgb", X[itr], y[itr], X[iva]) + fit_predict("lstm_app", X[itr], y[itr], X, mask_va, seed)) / 2
            sv = score(y[iva], pv, int(y[itr].mean() >= 0.5))
            margin = 1.96 * np.sqrt(sv["acc"] * (1 - sv["acc"]) / sv["n"])
            gate = bool(sv["edge"] > margin)
            pt = (fit_predict("xgb", X[itr], y[itr], X[te_c]) + fit_predict("lstm_app", X[itr], y[itr], X, te_c, seed)) / 2
            app["y"].append(y[te_c]); app["p"].append(pt); app["b"].append(np.full(te_c.sum(), int(y[itr].mean() >= 0.5)))
            app["gate"].append(np.full(te_c.sum(), gate)); app["years"].append((year, gate, sv["edge"], margin))
            if gate:
                probs_app[te_c] = pt

    res = {}
    for c, models in wf.items():
        res[c] = {}
        for m, d in models.items():
            if not d["y"]:
                continue
            y, p, b = (np.concatenate(d[k]) for k in ("y", "p", "b"))
            s = score(y, p, 0)
            s["base"] = float((y == b).mean())
            s["edge"] = s["acc"] - s["base"]
            res[c][m] = s
    out["walk_forward"] = res

    if app["y"]:
        y, p, b, g = (np.concatenate(app[k]) for k in ("y", "p", "b", "gate"))
        pred = (p >= 0.5).astype(int)
        on = g.astype(bool)
        out["fixed_quantai"] = {
            "windows": len(app["years"]), "windows_gated_on": int(sum(1 for _, gt, _, _ in app["years"] if gt)),
            "days_on": int(on.sum()), "days_total": int(len(y)),
            "edge_when_on": float((pred[on] == y[on]).mean() - (y[on] == b[on]).mean()) if on.any() else None,
            "edge_when_off": float((pred[~on] == y[~on]).mean() - (y[~on] == b[~on]).mean()) if (~on).any() else None,
            "n_on": int(on.sum()), "n_off": int((~on).sum()),
        }

    # --- Economics: long/flat, decide at close t, fill at close t+1, costs per side
    ret = raw["close"].pct_change().reindex(df.index)
    econ = {}
    for name, pr in (("ensemble_v2", probs_ens), ("fixed_quantai_gated", probs_app)):
        pos = (pr >= 0.5).astype(float).where(pr.notna())
        if name == "fixed_quantai_gated":
            pos = pos.where(pr.notna(), 1.0)  # gate off: the AI is ignored, so hold (buy and hold)
        held = pos.shift(2)
        mask = held.notna() & (df.index.year >= TEST_YEARS[0])
        net = (held * ret - held.diff().abs().fillna(0) * COST_PER_SIDE)[mask]
        econ[name] = {"sharpe": sharpe(net), "bh_sharpe": sharpe(ret[mask]),
                      "time_in_market": float(held[mask].mean())}
    out["economics"] = econ
    return out


# ------------------------------------------------------------------ main
def mean_ci(vals):
    from scipy.stats import t
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)])
    m, s = v.mean(), v.std(ddof=1)
    h = t.ppf(0.975, len(v) - 1) * s / np.sqrt(len(v))
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "n": int(len(v))}


def main():
    spy = load("SPY")
    universe, skipped = pick_universe()
    print(f"universe: {len(universe)} stocks (skipped {len(skipped)} without data back to 2010)", flush=True)
    tasks = [(u["symbol"], load(u["symbol"]), spy) for u in universe]
    results = []
    with ProcessPoolExecutor(max_workers=min(14, os.cpu_count() or 4)) as ex:
        for r in ex.map(run_stock, tasks):
            results.append(r)
            print(f"  done {r['symbol']} ({len(results)}/{len(tasks)})", flush=True)

    agg = {"isolation": {}, "walk_forward": {}, "pt_significant_share": {}, "fixed_quantai": {}, "economics": {}}
    for kind in ("xgb", "logit"):
        agg["isolation"][kind] = {p: {"edge": mean_ci([r["isolation"][kind][p]["edge"] for r in results]),
                                      "auc": mean_ci([r["isolation"][kind][p]["auc"] for r in results])}
                                  for p in ("P0_in_sample", "P1a_split_no_purge", "P1b_split_purged")}
    for c in ("v1_raw", "v2_raw", "v2_excess"):
        agg["walk_forward"][c] = {}
        for m in results[0]["walk_forward"][c]:
            rs = [r["walk_forward"][c][m] for r in results if m in r["walk_forward"].get(c, {})]
            agg["walk_forward"][c][m] = {
                "edge": mean_ci([s["edge"] for s in rs]), "auc": mean_ci([s["auc"] for s in rs]),
                "share_positive_edge": float(np.mean([s["edge"] > 0 for s in rs])),
                "share_pt_p_below_05": float(np.mean([s["pt_p"] < 0.05 for s in rs])),
                "up_rate": float(np.mean([s["up_rate"] for s in rs]))}
    fq = [r["fixed_quantai"] for r in results if "fixed_quantai" in r]
    agg["fixed_quantai"] = {
        "windows": int(sum(f["windows"] for f in fq)), "windows_gated_on": int(sum(f["windows_gated_on"] for f in fq)),
        "edge_when_on": mean_ci([f["edge_when_on"] for f in fq if f["edge_when_on"] is not None]),
        "edge_when_off": mean_ci([f["edge_when_off"] for f in fq if f["edge_when_off"] is not None]),
        "stocks_ever_on": int(sum(1 for f in fq if f["windows_gated_on"] > 0))}
    for name in ("ensemble_v2", "fixed_quantai_gated"):
        agg["economics"][name] = {
            "sharpe_minus_bh": mean_ci([r["economics"][name]["sharpe"] - r["economics"][name]["bh_sharpe"] for r in results]),
            "share_beating_bh": float(np.mean([r["economics"][name]["sharpe"] > r["economics"][name]["bh_sharpe"] for r in results])),
            "time_in_market": float(np.mean([r["economics"][name]["time_in_market"] for r in results]))}

    json.dump({"config": {"start": START, "end": END, "horizon": HORIZON, "test_years": [TEST_YEARS[0], TEST_YEARS[-1]],
                          "seed": SEED, "n_stocks": len(universe), "cost_per_side": COST_PER_SIDE},
               "universe": universe, "skipped": skipped, "aggregate": agg, "per_stock": results},
              open(os.path.join(HERE, "results_v2.json"), "w"), indent=1)
    make_figures(results, agg)
    print("done", flush=True)


def make_figures(results, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grey, accent = "#1d3a6e", "#9aa5ba", "#b4462e"

    # Figure 1: mean edge (with 95% CI across stocks) as each evaluation choice is removed
    steps = [("In-sample\n(train = test)", agg["isolation"]["xgb"]["P0_in_sample"]["edge"]),
             ("80/20 split,\nno purge", agg["isolation"]["xgb"]["P1a_split_no_purge"]["edge"]),
             ("80/20 split,\npurged", agg["isolation"]["xgb"]["P1b_split_purged"]["edge"]),
             ("Walk-forward,\nprice features", agg["walk_forward"]["v1_raw"]["xgb"]["edge"]),
             ("Walk-forward,\nstationary", agg["walk_forward"]["v2_raw"]["xgb"]["edge"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, (lab, e) in enumerate(steps):
        ax.bar(i, e["mean"] * 100, color=accent if i == 0 else ink, width=0.6)
        ax.errorbar(i, e["mean"] * 100, yerr=[[(e["mean"] - e["lo"]) * 100], [(e["hi"] - e["mean"]) * 100]],
                    color="#333", capsize=4, lw=1)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s for s, _ in steps], fontsize=8.5)
    ax.set_ylabel("XGBoost edge over\nmajority baseline (pp)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_isolation.png"), dpi=200)

    # Figure 2: distribution across the 50 stocks of walk-forward edge, by model (stationary features, raw label)
    models = [("logit", "Logistic"), ("xgb", "XGBoost"), ("lstm_app", "LSTM\n(original)"), ("lstm", "LSTM\n(proper)"), ("ens", "XGB+LSTM")]
    data = [[r["walk_forward"]["v2_raw"][m]["edge"] * 100 for r in results] for m, _ in models]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": accent}, boxprops={"color": ink},
               whiskerprops={"color": ink}, capprops={"color": ink})
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=7, color=grey, zorder=3)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([l for _, l in models], fontsize=8.5)
    ax.set_ylabel("Walk-forward edge (pp)\n50 stocks, 2016-2026", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_models.png"), dpi=200)

    # Figure 3: AUC by label
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (c, lab) in enumerate((("v2_raw", "Raw direction"), ("v2_excess", "Beat SPY"))):
        d = [r["walk_forward"][c]["xgb"]["auc"] for r in results]
        ax.scatter(np.full(len(d), i) + rng.uniform(-0.1, 0.1, len(d)), d, s=9, color=grey)
        m = np.mean(d)
        ax.plot([i - 0.25, i + 0.25], [m, m], color=accent, lw=2)
    ax.axhline(0.5, color="#333", lw=0.8, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw direction", "Beat SPY"], fontsize=9)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("XGBoost walk-forward AUC", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_label_auc.png"), dpi=200)


if __name__ == "__main__":
    sys.exit(main())
