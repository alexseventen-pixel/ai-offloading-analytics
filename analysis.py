"""Statistical analysis mirroring thesis Chapter 4 (descriptives, Pearson r, group comparison)."""
import numpy as np
import pandas as pd
from scipy import stats

COG_VARS = {"attention_index": "Attention index (0-100)", "luria_trial1": "Luria Trial 1 (/10)",
            "luria_delayed": "Luria delayed (/10)", "schulte_seconds": "Schulte time (s, lower=better)",
            "digit_span_total": "Digit Span total", "ct_total": "Critical thinking total (/8)"}


def correlations(df: pd.DataFrame, x="offloading_score") -> list[dict]:
    rows = []
    for col, label in COG_VARS.items():
        d = df[[x, col]].dropna()
        if len(d) < 3:
            continue
        r, p = stats.pearsonr(d[x], d[col])
        rho, _ = stats.spearmanr(d[x], d[col])
        strength = "strong" if abs(r) >= .5 else "moderate" if abs(r) >= .3 else "weak"
        rows.append({"variable": label, "n": len(d), "r": round(r, 3), "rho": round(rho, 3), "p": p,
                     "interpretation": f"{strength} {'negative' if r < 0 else 'positive'}"
                                       f"{'' if p < .05 else ' (n.s.)'}"})
    return rows


def critical_r(n: int, alpha=0.05) -> float:
    """Minimum |r| for p < alpha (two-tailed); thesis used 0.576 for n=12."""
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return round(float(t / np.sqrt(n - 2 + t ** 2)), 3)


def group_comparison(df: pd.DataFrame) -> list[dict]:
    """Low vs Higher AI-use groups (thesis Table 4.3 / 4.8) with Welch t-test and Cohen's d."""
    rows = []
    for col, label in [("offloading_score", "AI Offloading Score (1-5)")] + list(COG_VARS.items()):
        a = df.loc[df.ai_use_group == "Low", col].dropna()
        b = df.loc[df.ai_use_group == "Higher", col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"indicator": label, "low": f"{a.mean():.2f} (SD {a.std():.2f})",
                     "higher": f"{b.mean():.2f} (SD {b.std():.2f})", "n_low": len(a), "n_high": len(b),
                     "d": round(float((b.mean() - a.mean()) / sd), 2) if sd else 0.0, "p": p})
    return rows


def behaviour_profile(df: pd.DataFrame) -> list[dict]:
    """Percent indicators per group, same wording as thesis Table 4.3."""
    defs = [("Ask AI first - Often/Always", "q8_ask_ai_first", lambda s: s >= 4),
            ("Copy AI directly - Sometimes+", "q9_copy_directly", lambda s: s >= 3),
            ("Rewrite AI output - Often/Always", "q10_rewrite_own_words", lambda s: s >= 4),
            ("Check AI correctness - Often/Always", "q11_check_correctness", lambda s: s >= 4),
            ("Feel AI-independent - Prob/Def yes", "q12_could_do_without_ai", lambda s: s >= 4)]
    out = []
    for label, col, f in defs:
        row = {"indicator": label}
        for g in ("Low", "Higher"):
            s = df.loc[df.ai_use_group == g, col].dropna()
            row[g] = f"{100 * f(s).mean():.0f}%" if len(s) else "-"
        out.append(row)
    return out


def missing_report(df: pd.DataFrame) -> list[dict]:
    m = df.isna().sum()
    return [{"column": c, "missing": int(v), "pct": round(100 * v / len(df), 1)} for c, v in m.items() if v > 0]
