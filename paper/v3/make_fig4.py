"""Figure 4: in-sample accuracy vs walk-forward edge as regularization increases (from regularization_results.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "regularization_results.json")))["summary"]
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

families = [
    ("XGBoost", ["xgb_d4_v2default", "xgb_d2", "xgb_stumps"], ["depth 4\n(v2)", "depth 2", "stumps"], "#1f5fa8"),
    ("Logistic", ["logit_C1", "logit_C0.1", "logit_C0.01", "logit_C0.001"], ["C=1", "C=0.1\n(v2)", "C=0.01", "C=0.001"], "#c4532d"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
for ax, (name, keys, labels, col) in zip(axes, families):
    x = range(len(keys))
    gap = [S[k]["gap_train_minus_wf"]["mean"] * 100 for k in keys]
    edge = [S[k]["wf_edge"]["mean"] * 100 for k in keys]
    lo = [S[k]["wf_edge"]["lo"] * 100 for k in keys]
    hi = [S[k]["wf_edge"]["hi"] * 100 for k in keys]
    ax.bar(x, gap, color=col, alpha=0.25, label="Train minus walk-forward accuracy (pp)")
    ax.errorbar(x, edge, yerr=[[e - l for e, l in zip(edge, lo)], [h - e for e, h in zip(edge, hi)]],
                fmt="o-", color=col, capsize=3, label="Walk-forward edge over baseline (pp)")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{name}: weaker → stronger regularization", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("percentage points", fontsize=9)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(HERE, "figures", "fig4_regularization.png"), dpi=220)
print("fig4 written")
