"""
How much of an ML trading signal's edge survives honest evaluation?

Reproduces the QuantAI prediction pipeline (features, XGBoost + LSTM ensemble,
5-day direction label) and evaluates it under three protocols:

  P0  in-sample      : train and score on the same window (what QuantAI's AI
                       backtest does when it trains on the backtest window)
  P1  single split   : chronological 80/20 split, no purge (QuantAI's train())
  P2  walk-forward   : expanding window, retrained yearly, 5-bar purge between
                       train and test, scored only on unseen years

Then turns the walk-forward probabilities into a long/flat strategy with a
one-bar execution delay and QuantAI's cost assumptions.

Run:  python experiment.py            (writes results/ and figures/)
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]  # QuantAI's DEFAULT_SYMBOLS
START, END = "2010-01-01", "2026-11-01"
HORIZON = 5                                        # QuantAI's predictor horizon
TEST_YEARS = list(range(2016, 2026 + 1))
COMMISSION, SLIPPAGE = 0.001, 0.0001               # QuantAI's backtester costs
COST_PER_SIDE = COMMISSION + SLIPPAGE

PRICE_LEVEL = ["open", "high", "low", "close", "sma_20", "ema_9", "ema_21", "obv", "atr_14", "macd", "macd_signal",
               "macd_histogram", "index_close"]


# ---------------------------------------------------------------- data
def load(symbol):
    cache = os.path.join(RES, f"ohlcv_{symbol}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    df = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(cache)
    return df


# ------------------------------------------------- QuantAI feature set
def indicators(df):
    """Same formulas as QuantAI backend/app/data/indicators.py (all causal)."""
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


def features(df, index_df):
    """Same extra features and label as QuantAI predictor.prepare_features()."""
    df = indicators(df)
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    for lag in (1, 2, 3):
        df[f"return_1_lag{lag}"] = df["return_1"].shift(lag)
    for w in (5, 10, 20):
        df[f"volatility_{w}"] = df["return_1"].rolling(w).std()
    df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["volume_momentum"] = df["volume"].pct_change(5)
    if index_df is not None:
        ix = pd.DataFrame(index=index_df.index)
        ix["index_close"] = index_df["close"]
        ix["index_return_1"] = index_df["close"].pct_change()
        ix["index_above_sma50"] = (index_df["close"] > index_df["close"].rolling(50).mean()).astype(int)
        df = df.join(ix, how="left")
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(float)
    df.loc[df["close"].shift(-HORIZON).isna(), "target"] = np.nan
    cols = ["rsi_14", "macd", "macd_signal", "macd_histogram", "stoch_rsi_k", "stoch_rsi_d", "bb_pct", "bb_width",
            "atr_14", "sma_20", "ema_9", "ema_21", "trend_up", "above_sma_50", "price_vs_sma20", "momentum_score",
            "obv", "open", "high", "low", "close", "volume", "return_1", "return_5", "return_10", "return_1_lag1",
            "return_1_lag2", "return_1_lag3", "volatility_5", "volatility_10", "volatility_20", "volume_sma_ratio",
            "volume_momentum", "index_close", "index_return_1", "index_above_sma50"]
    cols = [c for c in cols if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    return df, cols


# ------------------------------------------------------------ models
class LSTMNet(nn.Module):
    """QuantAI lstm_model.py: 2-layer LSTM, hidden 64, sigmoid head."""
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(d, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return torch.sigmoid(self.fc(o[:, -1, :]))


def seqs(X, L=10):
    idx = np.arange(L - 1, len(X))
    return np.stack([X[i - L + 1:i + 1] for i in idx]), idx


def fit_predict(Xtr, ytr, Xall_scaled_fn, test_mask, Xall):
    """Fit scaler, XGBoost (QuantAI hyperparameters) and LSTM (15 full-batch epochs) on train rows;
    return probabilities for test rows from xgb, lstm and their average."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xall_s = sc.transform(Xtr), sc.transform(Xall)
    clf = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
                            random_state=42, verbosity=0)
    clf.fit(Xtr_s, ytr)
    p_xgb = clf.predict_proba(Xall_s[test_mask])[:, 1]

    S, idx = seqs(Xtr_s)
    net = LSTMNet(Xtr_s.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    xt, yt = torch.tensor(S, dtype=torch.float32), torch.tensor(ytr[idx], dtype=torch.float32).unsqueeze(1)
    for _ in range(15):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy(net(xt), yt)
        loss.backward()
        opt.step()
    Sall, idx_all = seqs(Xall_s)
    with torch.no_grad():
        p_all = np.full(len(Xall_s), 0.5)
        p_all[idx_all] = net(torch.tensor(Sall, dtype=torch.float32)).numpy().ravel()
    p_lstm = p_all[test_mask]
    return {"xgb": p_xgb, "lstm": p_lstm, "ens": (p_xgb + p_lstm) / 2}


# ----------------------------------------------------------- metrics
def block_boot_ci(correct, base_correct, block=20, reps=1000, seed=0):
    """95% CI for accuracy minus majority-baseline accuracy (moving-block bootstrap)."""
    rng = np.random.default_rng(seed)
    diff = correct.astype(float) - base_correct.astype(float)
    n = len(diff)
    nb = int(np.ceil(n / block))
    stats = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, nb)
        sample = np.concatenate([diff[s:s + block] for s in starts])[:n]
        stats.append(sample.mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def score(y, p, y_train):
    pred = (p >= 0.5).astype(int)
    maj = 1 if y_train.mean() >= 0.5 else 0          # majority class learned from training data only
    base = np.full_like(y, maj)
    lo, hi = block_boot_ci(pred == y, base == y)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return {"n": int(len(y)), "acc": float(accuracy_score(y, pred)), "baseline_acc": float((base == y).mean()),
            "edge": float(accuracy_score(y, pred) - (base == y).mean()), "edge_ci": [lo, hi], "auc": float(auc),
            "pred_up_rate": float(pred.mean())}


def perf(daily_ret):
    r = daily_ret.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(sharpe), "max_dd": float(dd),
            "total": float(eq.iloc[-1] - 1)}


# --------------------------------------------------------------- run
def run_symbol(sym, raw, spy):
    df, cols = features(raw, None if sym == "SPY" else spy)
    out = {"symbol": sym, "rows": int(len(df)), "first": str(df.index[0].date()), "last": str(df.index[-1].date())}
    lab = df["target"].notna().values
    X, y = df[cols].values, df["target"].values

    variants = {"quantai_features": cols, "stationary_only": [c for c in cols if c not in PRICE_LEVEL]}
    for vname, vcols in variants.items():
        Xv = df[vcols].values
        res = {}
        # P1: QuantAI single split (80/20, no purge) on the full labelled history
        L = np.where(lab)[0]
        cut = L[int(len(L) * 0.8)]
        tr = lab & (np.arange(len(df)) < cut)
        te = lab & (np.arange(len(df)) >= cut)
        p = fit_predict(Xv[tr], y[tr], None, te, Xv)
        res["P1_single_split"] = {m: score(y[te], p[m], y[tr]) for m in p}
        # P0: in-sample - same rows for training and scoring (the AI-backtest leak)
        p = fit_predict(Xv[te], y[te], None, te, Xv)
        res["P0_in_sample"] = {m: score(y[te], p[m], y[te]) for m in p}
        # P2: purged expanding walk-forward, retrain at the start of each test year
        probs = pd.Series(np.nan, index=df.index)
        ytrs, yy, pp = [], [], {m: [] for m in ("xgb", "lstm", "ens")}
        for yr in TEST_YEARS:
            te = lab & (df.index.year == yr)
            if te.sum() == 0:
                continue
            first = np.where(te)[0][0]
            tr = lab & (np.arange(len(df)) < first - HORIZON)   # purge: drop labels that overlap the test year
            if tr.sum() < 500:
                continue
            p = fit_predict(Xv[tr], y[tr], None, te, Xv)
            yy.append(y[te])
            ytrs.append(np.full(te.sum(), y[tr].mean()))
            for m in p:
                pp[m].append(p[m])
            if vname == "quantai_features":
                probs[te] = p["ens"]
        yy = np.concatenate(yy)
        trate = np.concatenate(ytrs)
        res["P2_walk_forward"] = {}
        for m in pp:
            pm = np.concatenate(pp[m])
            pred = (pm >= 0.5).astype(int)
            base = (trate >= 0.5).astype(int)                    # majority class of each training window
            lo, hi = block_boot_ci(pred == yy, base == yy)
            res["P2_walk_forward"][m] = {"n": int(len(yy)), "acc": float((pred == yy).mean()),
                                         "baseline_acc": float((base == yy).mean()),
                                         "edge": float((pred == yy).mean() - (base == yy).mean()),
                                         "edge_ci": [lo, hi], "auc": float(roc_auc_score(yy, pm)),
                                         "pred_up_rate": float(pred.mean())}
        out[vname] = res
        if vname == "quantai_features":
            out["_probs"] = probs
    # Strategy from walk-forward ensemble probabilities: decide at close t, trade at close t+1
    ret = raw["close"].pct_change()
    pr = out.pop("_probs").reindex(raw.index)
    strat = {}
    for thr in (0.5, 0.65):
        pos = (pr >= thr).astype(float).where(pr.notna())
        # Signal formed at close t, filled at close t+1 (QuantAI's one-bar delay), so the first
        # return earned is close t+1 -> close t+2.
        held = pos.shift(2)
        r = (held * ret).where(held.notna())
        turns = held.diff().abs().fillna(0)
        net = r - turns * COST_PER_SIDE
        mask = held.notna()
        strat[f"long_flat_p{thr}"] = {**perf(net[mask]), "time_in_market": float(held[mask].mean()),
                                      "trades_per_year": float(turns[mask].sum() / (mask.sum() / 252))}
        if thr == 0.5:
            out["_curve"] = (1 + net[mask].fillna(0)).cumprod()
            out["_bh"] = (1 + ret[mask].fillna(0)).cumprod()
    mask = pr.shift(2).notna()
    strat["buy_and_hold"] = {**perf(ret[mask]), "time_in_market": 1.0, "trades_per_year": 0.0}
    out["strategy"] = strat
    return out


def main():
    spy = load("SPY")
    results, curves = [], {}
    for sym in SYMBOLS:
        print(f"[{sym}] running...", flush=True)
        r = run_symbol(sym, load(sym), spy)
        curves[sym] = (r.pop("_curve"), r.pop("_bh"))
        results.append(r)
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump({"config": {"symbols": SYMBOLS, "start": START, "end": END, "horizon": HORIZON,
                              "test_years": [TEST_YEARS[0], TEST_YEARS[-1]], "cost_per_side": COST_PER_SIDE},
                   "results": results}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Figure 1: ensemble accuracy edge over majority baseline by protocol
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    prot = [("P0_in_sample", "In-sample\n(train = test window)"), ("P1_single_split", "80/20 split,\nno purge"),
            ("P2_walk_forward", "Purged\nwalk-forward")]
    w = 0.15
    for i, r in enumerate(results):
        vals = [r["quantai_features"][p]["ens"]["edge"] * 100 for p, _ in prot]
        ax.bar(np.arange(3) + (i - 2) * w, vals, w, label=r["symbol"])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in prot], fontsize=9)
    ax.set_ylabel("Accuracy minus majority\nbaseline (pp)", fontsize=9)
    ax.legend(fontsize=8, ncol=5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_edge_by_protocol.png"), dpi=200)
    # Figure 2: equity curves, walk-forward long/flat vs buy-and-hold
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.6), sharey=False)
    for ax, sym in zip(axes, SYMBOLS):
        c, bh = curves[sym]
        ax.plot(bh.index, bh.values, color="#9aa5ba", lw=1, label="Buy & hold")
        ax.plot(c.index, c.values, color="#1d3a6e", lw=1.2, label="Model long/flat")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mdates.YearLocator(5, month=1, day=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(sym, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_equity.png"), dpi=200)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
