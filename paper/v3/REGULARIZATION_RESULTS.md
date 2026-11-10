# Does restricting the model in-sample help out of sample?

Prompted by reader feedback on the v2 paper: the results suggest the model was overfitted, so restricting it in-sample might improve it out of sample.

**Setup:** the same 50 random S&P 500 stocks, stationary features, raw 5-day direction label and purged yearly walk-forward (2016–2026) as v2. Run it with `python regularization.py`; full output is in `regularization_results.json`.

| Model | Regularization | Train acc | Walk-forward acc | Baseline | Edge, pp [95% CI] | AUC | Train − WF gap (pp) | Stocks with edge > 0 |
|---|---|---|---|---|---|---|---|---|
| XGBoost, depth 4 (v2 default) | weak | 0.801 | 0.524 | 0.550 | −2.6 [−3.0, −2.2] | 0.504 | 27.7 | 2/50 |
| XGBoost, depth 2 | strong | 0.618 | 0.536 | 0.550 | −1.5 [−1.8, −1.1] | 0.501 | 8.2 | 4/50 |
| XGBoost stumps | very strong | 0.577 | 0.543 | 0.550 | −0.7 [−1.0, −0.4] | 0.500 | 3.3 | 11/50 |
| XGBoost stumps, 4 features | very strong | 0.569 | 0.544 | 0.550 | −0.6 [−0.9, −0.3] | 0.497 | 2.5 | 8/50 |
| Logistic, C=1 | weak | 0.582 | 0.530 | 0.550 | −2.1 [−2.5, −1.7] | 0.503 | 5.2 | 2/50 |
| Logistic, C=0.1 (v2 default) | moderate | 0.580 | 0.532 | 0.550 | −1.9 [−2.3, −1.5] | 0.504 | 4.8 | 1/50 |
| Logistic, C=0.01 | strong | 0.576 | 0.537 | 0.550 | −1.3 [−1.7, −0.9] | 0.504 | 3.9 | 8/50 |
| Logistic, C=0.001 | very strong | 0.565 | 0.547 | 0.550 | −0.3 [−0.5, −0.1] | 0.503 | 1.8 | 12/50 |
| Logistic L1, C=0.01 | strong, sparse | 0.559 | 0.546 | 0.550 | −0.4 [−0.7, −0.1] | 0.481 | 1.3 | 4/50 |
| Logistic, 4 features | moderate | 0.562 | 0.544 | 0.550 | −0.6 [−0.8, −0.4] | 0.500 | 1.8 | 4/50 |

## What it shows

1. **The reviewer is right about overfitting.** The default XGBoost fits its training data at 80% accuracy but scores 52% walk-forward, a 28-point gap. Restricting it to stumps shrinks the gap to 3 points.
2. **Restricting the model improves out-of-sample results, but only by removing harm.** The walk-forward edge rises from −2.6 to −0.7 points for XGBoost and from −1.9 to −0.3 for logistic regression, moving monotonically with regularization strength.
3. **It never produces skill.** No setting has a positive edge on average: every 95% interval is below zero. AUC stays at 0.48–0.50, the same as random ranking. The most regularized models get closer to the baseline because they increasingly predict "up" almost every day, which is what the baseline does.
4. **Conclusion for the paper:** overfitting explains why the original model did worse than the baseline. It doesn't explain the missing edge. With these price- and volume-based features, the 5-day direction of these stocks has close to no predictable component, so a model that fits less loses less but still doesn't win.

This could become a short section, "Is it just overfitting?", in a v3 of the paper.
