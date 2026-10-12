"""Matplotlib charts rendered to base64 PNG for the Flask templates."""
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db import LIKERT_LABELS

PALETTE = ["#2f6db5", "#e07a2f", "#3a9d5d", "#c4453c", "#7b5ea7", "#8c6d31", "#5aa9c4"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def likert_bar(df, col, title):
    labels = LIKERT_LABELS[col]
    counts = df[col].dropna().astype(int).value_counts().reindex(range(1, 6), fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, counts.values, color=PALETTE[0])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{100 * v / max(counts.sum(), 1):.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=15)
    return _png(fig)


def cat_barh(df, col, title):
    s = df[col].dropna().value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(s.index, s.values, color=PALETTE[1])
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    return _png(fig)


def offload_hist(df):
    fig, ax = plt.subplots(figsize=(5, 3))
    for g, c in (("Low", PALETTE[2]), ("Higher", PALETTE[3])):
        ax.hist(df.loc[df.ai_use_group == g, "offloading_score"].dropna(), bins=np.arange(1, 5.25, 0.25),
                alpha=0.65, color=c, label=f"{g} AI use")
    ax.axvline(2.0, ls="--", c="k", lw=1)
    ax.text(2.02, ax.get_ylim()[1] * 0.9, "thesis mean 2.0", fontsize=7)
    ax.set_xlabel("AI Offloading Score (1-5)")
    ax.set_title("Offloading composite by AI-use group", fontsize=10)
    ax.legend(fontsize=8)
    return _png(fig)


def scatter(df, y, ylabel):
    d = df[["offloading_score", y, "grade"]].dropna()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for g, c in ((8, PALETTE[0]), (9, PALETTE[1])):
        s = d[d.grade == g]
        ax.scatter(s.offloading_score, s[y], s=18, alpha=0.7, c=c, label=f"Grade {g}")
    if len(d) > 2:
        k, b = np.polyfit(d.offloading_score, d[y], 1)
        xs = np.linspace(d.offloading_score.min(), d.offloading_score.max(), 50)
        ax.plot(xs, k * xs + b, c="k", lw=1.5)
        ax.set_title(f"Offloading vs {ylabel}  (r = {d.offloading_score.corr(d[y]):.2f})", fontsize=10)
    ax.set_xlabel("AI Offloading Score")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return _png(fig)


def corr_heatmap(df):
    cols = ["q4_ai_frequency", "q8_ask_ai_first", "q9_copy_directly", "q10_rewrite_own_words",
            "q11_check_correctness", "q12_could_do_without_ai", "offloading_score", "screen_time_hours",
            "sleep_hours", "attention_index", "luria_trial1", "schulte_seconds", "ct_total"]
    short = ["Q4 freq", "Q8 ask 1st", "Q9 copy", "Q10 rewrite", "Q11 check", "Q12 indep", "Offload",
             "Screen h", "Sleep h", "Attention", "Luria T1", "Schulte s", "CT total"]
    c = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), short, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)), short, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation matrix", fontsize=10)
    return _png(fig)


def group_bars(df):
    m = df.groupby("ai_use_group")[["luria_trial1", "luria_delayed", "ct_total"]].mean().reindex(["Low", "Higher"])
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(3)
    for i, (g, c) in enumerate((("Low", PALETTE[2]), ("Higher", PALETTE[3]))):
        if g in m.index and not m.loc[g].isna().all():
            ax.bar(x + (i - 0.5) * 0.38, m.loc[g].values, 0.38, color=c, label=f"{g} AI use")
    ax.set_xticks(x, ["Luria T1 /10", "Luria delayed /10", "CT total /8"], fontsize=8)
    ax.set_title("Cognitive scores by AI-use group (cf. thesis Table 4.8)", fontsize=9)
    ax.legend(fontsize=8)
    return _png(fig)


def luria_curve(df, ref):
    cols = ["luria_trial1", "luria_trial2", "luria_trial3", "luria_delayed"]
    fig, ax = plt.subplots(figsize=(5, 3))
    for g, c in (("Low", PALETTE[2]), ("Higher", PALETTE[3])):
        s = df.loc[df.ai_use_group == g, cols].mean()
        ax.plot(["T1", "T2", "T3", "Delayed"], s.values, marker="o", c=c, label=f"Simulated - {g} AI use")
    if ref is not None:
        ax.plot(["T1", "T2", "T3", "Delayed"], ref[cols].mean().values, marker="s", ls="--", c="k",
                label="Thesis empirical (n=12)")
    ax.set_ylim(0, 10)
    ax.set_title("Luria 10 Words learning curve", fontsize=10)
    ax.legend(fontsize=7)
    return _png(fig)


def importance_bar(pairs, title, signed=False):
    names, vals = zip(*pairs[::-1]) if pairs else ([], [])
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    colors = [PALETTE[3] if v > 0 else PALETTE[0] for v in vals] if signed else PALETTE[0]
    ax.barh(names, vals, color=colors)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    if signed:
        ax.axvline(0, c="k", lw=0.8)
    return _png(fig)


def confusion(cm, title):
    fig, ax = plt.subplots(figsize=(3, 2.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Pred OK", "Pred risk"], fontsize=8)
    ax.set_yticks([0, 1], ["True OK", "True risk"], fontsize=8)
    ax.set_title(title, fontsize=8)
    return _png(fig)
