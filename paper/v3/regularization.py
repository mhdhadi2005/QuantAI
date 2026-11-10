"""
Regularization sweep: does restricting the model in-sample rescue out-of-sample skill?

Prompted by reviewer feedback on v2 ("the model was overfitted -- restrict it in-sample
and it may improve out of sample"). Same 50 stocks, same stationary features (v2), same
raw 5-day label and the same purged yearly walk-forward (2016-2026) as experiment_v2.py.

For each model we record, per stock:
  - train accuracy (in-sample fit)          -> how much the model memorises
  - walk-forward accuracy, baseline and edge -> whether anything survives out of sample
  - walk-forward AUC
The gap between train and walk-forward accuracy measures overfitting. If heavy
regularization closes the gap but the walk-forward edge stays at or below zero,
the problem is not overfitting alone: there is little signal to fit.

Run:  python regularization.py      (uses the cached prices in ../v2/data)
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
V2 = os.path.join(HERE, "..", "v2")
sys.path.insert(0, V2)
import experiment_v2 as e2  # noqa: E402

SMALL = ["return_5", "return_10", "momentum_score", "volatility_20"]  # 4 features


def models():
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    def lr(C, pen="l2"):
        return lambda: LogisticRegression(C=C, penalty=pen, solver="liblinear" if pen == "l1" else "lbfgs",
                                          max_iter=2000)

    def xg(depth, n, lr_, lam, mcw):
        return lambda: xgb.XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr_, subsample=0.8,
                                         colsample_bytree=0.8, reg_lambda=lam, min_child_weight=mcw,
                                         eval_metric="logloss", random_state=42, verbosity=0, n_jobs=1,
                                         tree_method="hist", device="cpu")

    # name: (factory, feature set, family, strength label)
    return {
        "logit_C1": (lr(1.0), "v2", "logit", "weak"),
        "logit_C0.1": (lr(0.1), "v2", "logit", "v2 default"),
        "logit_C0.01": (lr(0.01), "v2", "logit", "strong"),
        "logit_C0.001": (lr(0.001), "v2", "logit", "very strong"),
        "logit_L1_C0.01": (lr(0.01, "l1"), "v2", "logit", "strong, sparse"),
        "logit_small4": (lr(0.1), "small", "logit", "4 features"),
        "xgb_d4_v2default": (xg(4, 150, 0.03, 1.0, 1), "v2", "xgb", "v2 default"),
        "xgb_d2": (xg(2, 100, 0.03, 10.0, 20), "v2", "xgb", "strong"),
        "xgb_stumps": (xg(1, 50, 0.05, 10.0, 50), "v2", "xgb", "very strong"),
        "xgb_stumps_small4": (xg(1, 50, 0.05, 10.0, 50), "small", "xgb", "very strong, 4 features"),
    }


def run_stock(args):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sym, raw, spy = args
    df, _, v2 = e2.build(raw, spy)
    n = len(df)
    ar = np.arange(n)
    y = df["y_raw"].values
    lab = ~np.isnan(y)
    feats = {"v2": v2, "small": SMALL}
    acc = {m: {"y": [], "p": [], "b": [], "tr_acc": []} for m in models()}
    for year in e2.TEST_YEARS:
        in_year = df.index.year == year
        if not in_year.any():
            continue
        first = np.argmax(in_year)
        te = lab & in_year
        tr = lab & (ar < first - e2.HORIZON)
        if te.sum() == 0 or tr.sum() < 500:
            continue
        base_class = int(y[tr].mean() >= 0.5)
        for m, (factory, fs, _, _) in models().items():
            X = df[feats[fs]].values
            sc = StandardScaler().fit(X[tr])
            mdl = factory()
            mdl.fit(sc.transform(X[tr]), y[tr])
            p_tr = mdl.predict_proba(sc.transform(X[tr]))[:, 1]
            p_te = mdl.predict_proba(sc.transform(X[te]))[:, 1]
            acc[m]["tr_acc"].append(float(((p_tr >= 0.5) == y[tr]).mean()))
            acc[m]["y"].append(y[te]); acc[m]["p"].append(p_te); acc[m]["b"].append(np.full(te.sum(), base_class))
    out = {"symbol": sym}
    for m, d in acc.items():
        if not d["y"]:
            continue
        yy, pp, bb = (np.concatenate(d[k]) for k in ("y", "p", "b"))
        pred = (pp >= 0.5).astype(int)
        a = float((pred == yy).mean())
        base = float((yy == bb).mean())
        out[m] = {"train_acc": float(np.mean(d["tr_acc"])), "wf_acc": a, "wf_base": base, "wf_edge": a - base,
                  "wf_auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else 0.5,
                  "up_rate": float(pred.mean())}
    return out


def main():
    spy = e2.load("SPY")
    chosen, _ = e2.pick_universe()
    jobs = [(c["symbol"], e2.load(c["symbol"]), spy) for c in chosen]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        results = []
        for i, r in enumerate(ex.map(run_stock, jobs), 1):
            results.append(r)
            print(f"  done {r['symbol']} ({i}/{len(jobs)})", flush=True)
    summary = {}
    for m, (_, fs, fam, strength) in models().items():
        rows = [r[m] for r in results if m in r]
        row = {"family": fam, "features": fs, "regularization": strength, "stocks": len(rows)}
        for k in ("train_acc", "wf_acc", "wf_base", "wf_edge", "wf_auc", "up_rate"):
            row[k] = e2.mean_ci([x[k] for x in rows])
        row["gap_train_minus_wf"] = e2.mean_ci([x["train_acc"] - x["wf_acc"] for x in rows])
        row["stocks_edge_positive"] = int(sum(x["wf_edge"] > 0 for x in rows))
        summary[m] = row
    with open(os.path.join(HERE, "regularization_results.json"), "w") as f:
        json.dump({"per_stock": results, "summary": summary}, f, indent=1)
    print("\nmodel                 train   wf_acc  base    edge(pp) [95% CI]        AUC    gap(pp)  stocks>0")
    for m, s in summary.items():
        print(f"{m:21s} {s['train_acc']['mean']:.3f}  {s['wf_acc']['mean']:.3f}  {s['wf_base']['mean']:.3f}  "
              f"{s['wf_edge']['mean']*100:+5.1f} [{s['wf_edge']['lo']*100:+.1f},{s['wf_edge']['hi']*100:+.1f}]  "
              f"{s['wf_auc']['mean']:.3f}  {s['gap_train_minus_wf']['mean']*100:5.1f}   {s['stocks_edge_positive']}/50")


if __name__ == "__main__":
    main()
