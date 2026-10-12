"""Synthetic, thesis-aligned dataset generator.

Simulates a scaled-up replication (N=180) of the Master's thesis study
"Use of AI Tools and Cognitive Offloading among 8th and 9th Grade Students:
Associations with Attention and Critical Thinking in a Latvian International School".

Instruments mirrored from the thesis (Chapter 3, Attachment A):
  * 12-item questionnaire (Q1-Q12) on AI use; Q4, Q8, Q9, Q10r, Q11r, Q12r -> AI Offloading Score (1-5)
  * Luria's 10 Words Test (3 immediate trials + delayed recall, 0-10)
  * Schulte Tables (mean seconds over three 5x5 grids; lower = better attention)
  * Digit Span (forward + backward correctly repeated sequences)
  * Critical Thinking "Survival Story" fact-checking task: detection 0-4 + justification 0-4 = 0-8

Distributions are calibrated to the thesis sample (mostly "Rarely/Sometimes" AI use,
ChatGPT ~67%, explanation as main purpose, CT mean ~5.6/8, Luria T1 mean ~6.8, Schulte < 45 s),
while the AI-offloading -> cognition effect is injected at a moderate negative size
(literature: Gerlich 2025 r=-0.68, Tian & Zhang 2025 r=-0.35), so the hypothesis can be
tested with adequate power. THE DATA ARE SYNTHETIC and must not be reported as real findings.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 2026
N = 180
OUT = Path(__file__).parent / "data" / "ai_offloading_students.csv"

FREQ = ["Never", "Rarely", "Sometimes", "Often", "Very often"]          # Q4
LIKERT = ["Never", "Rarely", "Sometimes", "Often", "Always"]             # Q8-Q11
TIME = ["0 min", "1-10 min", "11-20 min", "21-30 min", "30+ min"]        # Q7
INDEP = ["Definitely no", "Probably no", "Not sure", "Probably yes", "Definitely yes"]  # Q12 (5 = most independent)
TOOLS = ["ChatGPT", "Gemini", "Copilot", "Translation AI", "Grammar AI"]
SUBJECTS = ["Languages", "Math", "Science", "Social studies", "Other"]
PURPOSES = ["Explanation", "Grammar correction", "Ideas / outline", "Translation", "Summary",
            "Full answer", "Solving tasks"]


def _ordinal(latent, cuts):
    """Map a continuous latent value to 1..5 using cut points."""
    return (np.digitize(latent, cuts) + 1).astype(int)


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = rng.choice([8, 9], n, p=[0.55, 0.45])
    gender = rng.choice(["Female", "Male"], n, p=[0.46, 0.54])          # thesis: 10 girls / 12 boys
    age = np.where(grade == 8, rng.choice([13, 14], n, p=[0.6, 0.4]), rng.choice([14, 15], n, p=[0.6, 0.4]))

    ability = rng.normal(0, 1, n)                     # latent academic ability (confounder noted in thesis 4.5)
    info_lit = 0.4 * ability + rng.normal(0, 0.9, n)  # information literacy (protective factor, Tian & Zhang)
    # latent offloading tendency: higher in grade 9, lower with information literacy
    off = rng.normal(0, 1, n) + 0.25 * (grade == 9) - 0.35 * info_lit

    q4 = _ordinal(off + rng.normal(0, 0.6, n), [-2.3, -0.2, 0.9, 1.8])            # mostly Rarely/Sometimes
    q7 = np.clip(_ordinal(off + rng.normal(0, 0.7, n), [-2.3, 0.1, 1.2, 2.0]), 1, 5)
    q7 = np.where(q4 == 1, 1, np.maximum(q7, 2))      # non-users spend 0 min
    q8 = _ordinal(off + rng.normal(0, 0.7, n), [-0.6, 0.4, 1.3, 2.1])          # ask AI first
    q9 = _ordinal(off + rng.normal(0, 0.7, n), [-0.2, 0.8, 1.6, 2.4])          # copy directly (rare)
    q10 = _ordinal(-off + rng.normal(0, 0.8, n), [-2.4, -1.6, -0.6, 0.2])      # rewrite (mostly often/always)
    q11 = _ordinal(-off + 0.4 * info_lit + rng.normal(0, 0.8, n), [-2.4, -1.4, -0.6, 0.0])  # check output
    q12 = _ordinal(-off + 0.4 * ability + rng.normal(0, 0.8, n), [-2.4, -1.4, -0.4, 0.8])   # could do without AI

    tool = rng.choice(TOOLS, n, p=[0.55, 0.18, 0.07, 0.10, 0.10])
    subject = rng.choice(SUBJECTS, n, p=[0.38, 0.24, 0.18, 0.14, 0.06])
    # purpose depends on offloading: high offloaders pick "Full answer"/"Solving tasks" more often
    p_base = np.array([0.36, 0.17, 0.14, 0.11, 0.10, 0.07, 0.05])
    purpose = []
    for o in off:
        w = p_base * np.exp(np.array([-0.3, 0, 0, 0, 0.1, 0.8, 0.8]) * o)
        purpose.append(rng.choice(PURPOSES, p=w / w.sum()))

    screen = np.clip(2.3 + 0.5 * off + rng.normal(0, 1.0, n), 0.3, 7.0).round(1)  # recreational h/day
    sleep = np.clip(8.4 - 0.25 * screen + rng.normal(0, 0.6, n), 5.5, 10.5).round(1)

    offload_score = (q4 + q8 + q9 + (6 - q10) + (6 - q11) + (6 - q12)) / 6.0
    oz = (offload_score - offload_score.mean()) / offload_score.std()

    # attention latent: ability + sleep - offloading - screen time (Leonhardt 2025, Sina 2024)
    att = 0.55 * ability - 0.40 * oz - 0.20 * (screen - 2.3) + 0.15 * (sleep - 8) + rng.normal(0, 0.6, n)
    luria1 = np.clip(np.round(6.8 + 1.4 * att + rng.normal(0, 0.8, n)), 2, 10)
    luria2 = np.clip(np.round(luria1 + 0.4 + rng.normal(0, 0.9, n)), 2, 10)
    luria3 = np.clip(np.round(luria2 - 0.1 + rng.normal(0, 0.9, n)), 2, 10)
    luria_d = np.clip(np.round(luria3 - 0.8 + 0.3 * att + rng.normal(0, 0.7, n)), 0, 10)
    schulte = np.clip(38 - 5.0 * att + rng.normal(0, 3.5, n), 24, 75).round(1)
    digit = np.clip(np.round(12 + 2.2 * att + rng.normal(0, 1.5, n)), 4, 22)

    # critical thinking latent: ability + info literacy - offloading (Gerlich 2025)
    ct = 0.55 * ability + 0.25 * info_lit - 0.45 * oz + rng.normal(0, 0.6, n)
    det = np.clip(np.round(3.0 + 0.9 * ct + rng.normal(0, 0.5, n)), 0, 4)
    jus = np.clip(np.minimum(det, np.round(2.6 + 1.0 * ct + rng.normal(0, 0.5, n))), 0, 4)

    df = pd.DataFrame({
        "student_id": [f"P{1001 + i}" for i in range(n)],
        "grade": grade, "gender": gender, "age": age,
        "main_ai_tool": tool, "main_subject": subject, "main_purpose": purpose,
        "q4_ai_frequency": q4, "q7_daily_ai_time": q7, "q8_ask_ai_first": q8,
        "q9_copy_directly": q9, "q10_rewrite_own_words": q10, "q11_check_correctness": q11,
        "q12_could_do_without_ai": q12,
        "screen_time_hours": screen, "sleep_hours": sleep,
        "luria_trial1": luria1.astype(int), "luria_trial2": luria2.astype(int),
        "luria_trial3": luria3.astype(int), "luria_delayed": luria_d.astype(int),
        "schulte_seconds": schulte, "digit_span_total": digit.astype(int),
        "ct_detection": det.astype(int), "ct_justification": jus.astype(int),
    })
    # realistic data-quality issues for the preprocessing module: ~3% missing answers
    for col in ["q11_check_correctness", "q12_could_do_without_ai", "screen_time_hours", "sleep_hours",
                "digit_span_total", "main_subject"]:
        idx = rng.choice(n, size=max(1, int(0.03 * n)), replace=False)
        df[col] = df[col].astype(object)
        df.loc[idx, col] = np.nan
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    d = generate()
    d.to_csv(OUT, index=False)
    print(f"saved {len(d)} rows -> {OUT}")
