# How to train a QuantAI model

A hands-on guide. Run the companion script and read along:

```bash
cd backend
python ../learn/train_step_by_step.py          # AAPL by default
python ../learn/train_step_by_step.py NVDA     # any symbol
```

It takes about a minute. The numbers quoted below are from an AAPL run on 4 Nov 2026; yours will differ slightly as new data arrives.

---

## The idea in one paragraph

Training means showing a model many past examples of *situation → outcome*, so it learns a rule that maps one to the other. In QuantAI, a **situation** is one trading day described by 28 numbers (features), and the **outcome** (the label) is whether the price was higher 5 trading days later. The model's job is to output a probability that the next 5 days go up. Everything else in this guide is about one question: *does that rule work on days the model has never seen?*

---

## Step 1 – Data

`get_ohlcv(symbol, "1d", days)` downloads daily bars from Yahoo Finance: open, high, low, close and volume. AAPL gives about 2,900 bars over 8 years.

## Step 2 – Features

`compute_all_indicators()` adds technical indicators. `AIPredictor.prepare_features()` turns them into 28 **stationary** features: RSI, Bollinger %B, MACD divided by price, recent returns, volatility, volume ratios, and the market index's return.

**Why stationary?** A tree model learns rules like "if close > 180 then…". When AAPL trades at 330, every day falls past the edge of anything it saw in training, and the rule breaks. Ratios and returns mean the same thing in 2016 and 2026. (The old QuantAI used raw prices; the working paper shows this was a major reason it failed.)

## Step 3 – The label and the baseline

The label is 1 if `close[t+5] > close[t]`. For AAPL, **58.8%** of days are "up".

That number is your **baseline**: a "model" that always says "up" is already right 58.8% of the time. A model scoring 57% looks better than a coin flip but is *worse than doing nothing clever*. Always compare against the baseline, never against 50%.

## Step 4 – Split by time, then purge

You need data the model has never seen to judge it. In finance that must be the **future**:

```
|<------------------- train (2016 → Aug 2024) ------------------->|gap|<-- test (Sep 2024 → Oct 2026) -->|
```

- **Never split randomly.** The script tries both. The random split showed **64.1%** accuracy, the time split **56.5%**. The random number is fake: today's label (next 5 days) overlaps almost entirely with tomorrow's, so a random split lets the model see the answer through a neighbour. This is called *leakage*.
- **Purge the gap.** The last 5 training days have labels that look into the test period, so they're dropped.

## Step 5 – Train, and train something simple too

The script trains QuantAI's XGBoost (150 shallow trees, learning rate 0.03) and a plain logistic regression:

| Model | Accuracy | Baseline | Edge | AUC |
|---|---|---|---|---|
| XGBoost | 56.5% | 56.5% | +0.0 pp | 0.546 |
| Logistic regression | 53.3% | 56.5% | −3.1 pp | 0.452 |

**Always fit a simple model next to the complex one.** If the simple one matches it, the complexity isn't buying anything. Here neither beats the baseline.

**Scaling:** `StandardScaler` is fitted on training rows only, then applied to test rows. Fitting it on everything would leak test-period averages into training.

## Step 6 – Read the result properly

The confusion matrix shows what really happened:

```
               said DOWN  said UP
  went DOWN           67      168
  went UP             67      238
```

The model said "up" 406 of 540 times, so it mostly copies the baseline. Three habits:

1. **Edge** = accuracy − baseline. This is the number that matters.
2. **Confidence interval.** With 540 test days, accuracy is only known to about ±4 points (52.3%–60.7%). The baseline sits inside that range, so no edge can be claimed.
3. **AUC.** It asks whether the model's probabilities rank up-days above down-days. 0.50 is chance; real daily signals in finance are often only 0.52–0.55.

## Step 7 – Walk-forward

One test window can mislead you, so retrain every year on all earlier data and test on the next year:

| Year | Edge |
|---|---|
| 2019 | −23.0 pp |
| 2020 | −2.8 pp |
| 2021 | +2.4 pp |
| 2022 | +4.0 pp |
| 2023 | −11.2 pp |
| 2024 | −2.8 pp |
| 2025 | +0.4 pp |
| 2026 | +5.8 pp |

The average edge is **−3.4 pp**. If you had tested only on 2026, you'd believe you had a +5.8-point edge. That's why QuantAI's backtester now retrains walk-forward.

## Step 8 – How much data you need

| History | Train / test days | Width of 95% interval |
|---|---|---|
| 2 years | 375 / 96 | ±10 pp (20 pp wide) |
| 8 years | 1,582 / 397 | ±5 pp (10 pp wide) |

The app trains on `LOOKBACK_DAYS × 2` = about **2 years** by default. When all 20 default symbols were trained this way, the edges ranged from −12 to +17 points, and 11 of 20 models "passed" the beats-baseline check. With ±10-point error bars, most of those passes are luck.

**Fix:** set `LOOKBACK_DAYS=1460` (or more) in `backend/.env` before training. This also makes the app fetch more history for charts and signals, which is slower but harmless.

## Step 9 – Training in the app

The whole pipeline above is one call:

```python
from app.models.predictor import AIPredictor
model = AIPredictor("AAPL")         # persist=True: saves to backend/models_saved/
metrics = model.train(df)           # df = compute_all_indicators(get_ohlcv(...)).dropna()
model.has_edge()                    # True only if edge > its 95% margin of error
```

In the running app the same thing happens from:

- the **Train All Models** button on the AI page
- `POST /api/ai/train`
- the daily scheduler (every `MODEL_RETRAIN_INTERVAL_HOURS`)

Each symbol saves three files: `<SYMBOL>_h5_v2_clf.pkl` (the XGBoost + LSTM models), `_scaler.pkl` and `_meta.pkl` (metrics). Live trading uses a model's predictions only when `has_edge()` is true: the edge over the baseline must be larger than its 95% margin of error (`edge_margin_95`), not just positive.

**What that means in practice:** retrained on 4 years (`LOOKBACK_DAYS=730`), each model is tested on about 248 days, so its margin is about ±6 points. Of the 20 default symbols, 13 had a positive edge, but only NFLX (+17.7) and DIS (+6.5) cleared their margin. With 20 models tested, one or two clearing it by chance is roughly what you'd expect, so even those deserve suspicion until a walk-forward test confirms them.

---

## Exercises (in order of difficulty)

1. **Another stock.** Run the script for `NVDA`, `SPY` and `RELIANCE.NS`. Does any have a positive average walk-forward edge?
2. **Another horizon.** Change `HORIZON = 5` to `1` and then `20`. Which horizon gives the highest AUC? Why might longer horizons look better? (Hint: count how many *independent* test periods you really have.)
3. **A fairer label.** Change the label to "beats SPY over the next 5 days". The baseline drops to about 50%, so any edge is real. How do you build it in `prepare_features()`?
4. **Feature importance.** After training, sort `dict(zip(X.columns, model.feature_importances_))`. Are the top features stable from one walk-forward year to the next? If not, the model is fitting noise.
5. **Predict volatility instead.** Make the label next week's realised volatility (a regression target). Compare against "next week = this week". Volatility is genuinely predictable, so this is where you'll first see a real edge.
6. **Go cross-sectional.** Train one model across 50+ stocks and predict which ones will *outperform the others* next month. This is the setup used in academic research (Gu, Kelly & Xiu, 2020), and the natural next project.

## Checklist before you trust any trained model

- [ ] Split by time, never randomly, and purge overlapping labels
- [ ] Scaler and any tuning fitted on training data only
- [ ] Edge reported against the majority-class baseline
- [ ] Confidence interval excludes the baseline
- [ ] A simple model trained alongside for comparison
- [ ] Positive average edge across walk-forward years, not just one
- [ ] Enough test days (hundreds, not dozens)
