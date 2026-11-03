"""
Training a QuantAI model, step by step.

Run from the backend folder:
    cd backend
    python ../learn/train_step_by_step.py

Each step prints what it did and what to notice. Read it alongside
learn/TRAINING_GUIDE.md, which explains the ideas behind each step.
"""
import logging
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from app.data.data_loader import get_ohlcv
from app.data.indicators import compute_all_indicators
from app.models.predictor import AIPredictor

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
HORIZON = 5


def step(n, title):
    print(f"\n{'=' * 70}\nSTEP {n}: {title}\n{'=' * 70}")


def ci95(acc, n):
    """Rough 95% interval for an accuracy measured on n test days (ignores overlap, so it is optimistic)."""
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)
    return acc - half, acc + half


def make_xgb():
    # Same settings QuantAI uses
    return xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                             eval_metric="logloss", random_state=42, verbosity=0)


def fit_score(X_tr, y_tr, X_te, y_te, model):
    sc = StandardScaler().fit(X_tr)                   # fit the scaler on training rows only
    model.fit(sc.transform(X_tr), y_tr)
    p = model.predict_proba(sc.transform(X_te))[:, 1]
    pred = (p >= 0.5).astype(int)
    majority = int(y_tr.mean() >= 0.5)
    return {"acc": accuracy_score(y_te, pred), "base": float((y_te == majority).mean()),
            "auc": roc_auc_score(y_te, p) if y_te.nunique() > 1 else 0.5, "pred": pred, "p": p}


# ----------------------------------------------------------------------
step(1, "Get price data")
raw = get_ohlcv(SYMBOL, "1d", 8 * 365)
print(f"{SYMBOL}: {len(raw)} daily bars from {raw.index[0].date()} to {raw.index[-1].date()}")
print(raw.tail(3).round(2).to_string())
print("-> Each row is one trading day: open, high, low, close, volume.")

# ----------------------------------------------------------------------
step(2, "Turn prices into features")
df = compute_all_indicators(raw).dropna()
predictor = AIPredictor(SYMBOL, horizon=HORIZON, persist=False)
X, y = predictor.prepare_features(df)
print(f"{X.shape[1]} features per day, e.g.: {', '.join(X.columns[:8])}, ...")
print(X.tail(2).round(3).iloc[:, :6].to_string())
print("-> All features are returns, ratios or oscillators. None is a raw price,")
print("   so a model trained in 2019 still makes sense when prices are 3x higher.")

# ----------------------------------------------------------------------
step(3, "Define the label and the baseline")
print(f"Label = 1 if the close {HORIZON} trading days later is higher, else 0.")
print(f"Usable days: {len(y)}.  Share of 'up' labels: {y.mean():.1%}")
print(f"-> Baseline: always guessing 'up' would already be right {max(y.mean(), 1 - y.mean()):.1%} of the time.")
print("   A model is only useful if it beats THAT, not 50%.")

# ----------------------------------------------------------------------
step(4, "Split by time, and purge the overlap")
cut = int(len(X) * 0.8)
X_tr, y_tr = X.iloc[:cut - HORIZON], y.iloc[:cut - HORIZON]
X_te, y_te = X.iloc[cut:], y.iloc[cut:]
print(f"Train: {X_tr.index[0].date()} -> {X_tr.index[-1].date()}  ({len(X_tr)} days)")
print(f"Purged gap: {HORIZON} days (their labels look into the test period)")
print(f"Test:  {X_te.index[0].date()} -> {X_te.index[-1].date()}  ({len(X_te)} days)")

print("\nWhy not a random split? Watch what it does:")
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
r_tr, r_te = idx[:cut], idx[cut:]
rand = fit_score(X.iloc[r_tr], y.iloc[r_tr], X.iloc[r_te], y.iloc[r_te], make_xgb())
timed = fit_score(X_tr, y_tr, X_te, y_te, make_xgb())
print(f"  random split: accuracy {rand['acc']:.1%}  (baseline {rand['base']:.1%})  <- looks great")
print(f"  time split:   accuracy {timed['acc']:.1%}  (baseline {timed['base']:.1%})  <- the honest number")
print("-> Neighbouring days share 4 of their 5 future days, so a random split lets the")
print("   model peek at the answer through its neighbours. Always split by time.")

# ----------------------------------------------------------------------
step(5, "Train two models: XGBoost and a simple logistic regression")
logit = fit_score(X_tr, y_tr, X_te, y_te, LogisticRegression(max_iter=2000, C=0.1))
for name, r in (("XGBoost", timed), ("Logistic regression", logit)):
    lo, hi = ci95(r["acc"], len(y_te))
    print(f"  {name:20} accuracy {r['acc']:.1%} [95% CI {lo:.1%}-{hi:.1%}]  "
          f"baseline {r['base']:.1%}  edge {(r['acc'] - r['base']) * 100:+.1f}pp  AUC {r['auc']:.3f}")
print("-> Always train a simple model next to the fancy one. If the simple one does")
print("   as well, the complexity is not buying you anything.")

# ----------------------------------------------------------------------
step(6, "Read the evaluation properly")
cm = confusion_matrix(y_te, timed["pred"], labels=[0, 1])
print("Confusion matrix (rows = what happened, columns = what the model said):")
print(f"               said DOWN  said UP")
print(f"  went DOWN    {cm[0, 0]:>9}  {cm[0, 1]:>7}")
print(f"  went UP      {cm[1, 0]:>9}  {cm[1, 1]:>7}")
lo, hi = ci95(timed["acc"], len(y_te))
print(f"-> The 95% interval for accuracy is {lo:.1%} to {hi:.1%}. If the baseline")
print(f"   ({timed['base']:.1%}) is inside it, you cannot claim the model beats it.")
print("   AUC near 0.50 means the model's probabilities don't rank up-days above down-days.")

# ----------------------------------------------------------------------
step(7, "Walk-forward: retrain every year, test on the next year")
rows = []
for year in sorted(set(X.index.year))[3:]:
    te = X.index.year == year
    first = np.argmax(te)
    tr = np.arange(len(X)) < first - HORIZON
    if tr.sum() < 500 or te.sum() < 50:
        continue
    r = fit_score(X[tr], y[tr], X[te], y[te], make_xgb())
    rows.append((year, int(tr.sum()), int(te.sum()), r["acc"], r["base"], r["auc"]))
    print(f"  {year}: trained on {tr.sum():>4} days, tested on {te.sum():>3}  "
          f"accuracy {r['acc']:.1%}  baseline {r['base']:.1%}  edge {(r['acc'] - r['base']) * 100:+5.1f}pp  AUC {r['auc']:.2f}")
if rows:
    edges = [(a - b) * 100 for _, _, _, a, b, _ in rows]
    print(f"-> Average edge across years: {np.mean(edges):+.1f}pp. The year-to-year swings show how")
    print("   much a single test window can mislead you.")

# ----------------------------------------------------------------------
step(8, "More data, smaller error bars")
for years in (2, 8):
    sub = df[df.index >= df.index[-1] - pd.Timedelta(days=365 * years)]
    Xs, ys = AIPredictor(SYMBOL, horizon=HORIZON, persist=False).prepare_features(sub)
    c = int(len(Xs) * 0.8)
    r = fit_score(Xs.iloc[:c - HORIZON], ys.iloc[:c - HORIZON], Xs.iloc[c:], ys.iloc[c:], make_xgb())
    lo, hi = ci95(r["acc"], len(Xs) - c)
    print(f"  {years} years of history: {c - HORIZON:>4} train / {len(Xs) - c:>3} test days -> accuracy {r['acc']:.1%}, "
          f"95% CI width {(hi - lo) * 100:.0f}pp")
print("-> The app trains on ~2 years by default (LOOKBACK_DAYS x 2). Test sets that small")
print("   can't tell a real edge from luck. Set LOOKBACK_DAYS higher in backend/.env.")

# ----------------------------------------------------------------------
step(9, "Train exactly like the app does (XGBoost + LSTM ensemble)")
app_model = AIPredictor(SYMBOL, horizon=HORIZON, persist=False)
m = app_model.train(df)
print(f"  accuracy {m['accuracy']:.1%}  baseline {m['baseline_accuracy']:.1%}  "
      f"edge {m['edge_vs_baseline'] * 100:+.1f}pp ±{m['edge_margin_95'] * 100:.1f}  AUC {m['auc']:.3f}")
print(f"  live trading would {'USE' if app_model.has_edge() else 'IGNORE'} this model "
      f"(its edge must be bigger than its margin of error).")
print("-> This is one call: AIPredictor(symbol).train(df). In the app the same thing runs")
print("   from the 'Train All Models' button, POST /api/ai/train, or the daily scheduler,")
print("   and the result is saved to backend/models_saved/<SYMBOL>_h5_v2_*.pkl")
print("\nDone. Now try the exercises at the end of learn/TRAINING_GUIDE.md.")
