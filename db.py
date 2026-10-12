"""SQLite persistence layer (CRUD) + derived thesis indicators.

Derived columns (recomputed on every insert/update):
  offloading_score  mean of Q4, Q8, Q9, (6-Q10), (6-Q11), (6-Q12)  -> 1..5, higher = more offloading (thesis 4.3)
  ai_use_group      "Low" (Q4 <= Rarely) vs "Higher" (Sometimes+)  -> thesis Table 4.3 split
  luria_mean        mean of the three immediate Luria trials
  ct_total          ct_detection + ct_justification -> 0..8
  attention_index   0..100 composite of Luria T1, Luria delayed, Schulte (inverted), Digit Span
  low_attention     1 if attention_index < LOW_ATTENTION_CUT
  low_ct            1 if ct_total <= LOW_CT_CUT (below the thesis "moderate" band 5-7)
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "ai_offloading.db"
CSV_PATH = BASE / "data" / "ai_offloading_students.csv"

LOW_ATTENTION_CUT = 55.0
LOW_CT_CUT = 4

CATEGORICAL = {
    "gender": ["Female", "Male"],
    "main_ai_tool": ["ChatGPT", "Gemini", "Copilot", "Translation AI", "Grammar AI"],
    "main_subject": ["Languages", "Math", "Science", "Social studies", "Other"],
    "main_purpose": ["Explanation", "Grammar correction", "Ideas / outline", "Translation", "Summary",
                     "Full answer", "Solving tasks"],
}
LIKERT_LABELS = {
    "q4_ai_frequency": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "q7_daily_ai_time": ["0 min", "1-10 min", "11-20 min", "21-30 min", "30+ min"],
    "q8_ask_ai_first": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "q9_copy_directly": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "q10_rewrite_own_words": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "q11_check_correctness": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "q12_could_do_without_ai": ["Definitely no", "Probably no", "Not sure", "Probably yes", "Definitely yes"],
}
QUESTION_TEXT = {
    "q4_ai_frequency": "Q4 How often do you use AI for homework?",
    "q7_daily_ai_time": "Q7 Time per day using AI for homework",
    "q8_ask_ai_first": "Q8 When homework is difficult, ask AI first?",
    "q9_copy_directly": "Q9 Copy AI text directly into your work?",
    "q10_rewrite_own_words": "Q10 Rewrite AI answers in your own words?",
    "q11_check_correctness": "Q11 Check whether the AI answer is correct?",
    "q12_could_do_without_ai": "Q12 Could you do most homework without AI?",
}
NUMERIC = {  # name: (min, max, step)
    "grade": (8, 9, 1), "age": (12, 16, 1),
    "screen_time_hours": (0, 12, 0.1), "sleep_hours": (3, 12, 0.1),
    "luria_trial1": (0, 10, 1), "luria_trial2": (0, 10, 1), "luria_trial3": (0, 10, 1),
    "luria_delayed": (0, 10, 1), "schulte_seconds": (10, 180, 0.1), "digit_span_total": (0, 28, 1),
    "ct_detection": (0, 4, 1), "ct_justification": (0, 4, 1),
}
INPUT_FIELDS = ["grade", "gender", "age"] + list(CATEGORICAL)[1:] + list(LIKERT_LABELS) + \
    [k for k in NUMERIC if k not in ("grade", "age")]
DERIVED = ["offloading_score", "ai_use_group", "luria_mean", "ct_total", "attention_index",
           "low_attention", "low_ct"]


def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def derive(rec: dict) -> dict:
    """Compute thesis indicators for one record (missing items are skipped, not zero-filled)."""
    r = dict(rec)
    items = [_num(r.get("q4_ai_frequency")), _num(r.get("q8_ask_ai_first")), _num(r.get("q9_copy_directly"))]
    items += [None if _num(r.get(k)) is None else 6 - _num(r.get(k))
              for k in ("q10_rewrite_own_words", "q11_check_correctness", "q12_could_do_without_ai")]
    items = [i for i in items if i is not None]
    r["offloading_score"] = round(sum(items) / len(items), 3) if len(items) >= 4 else None
    q4 = _num(r.get("q4_ai_frequency"))
    r["ai_use_group"] = None if q4 is None else ("Low" if q4 <= 2 else "Higher")
    lur = [_num(r.get(k)) for k in ("luria_trial1", "luria_trial2", "luria_trial3")]
    lur = [v for v in lur if v is not None]
    r["luria_mean"] = round(sum(lur) / len(lur), 2) if lur else None
    d, j = _num(r.get("ct_detection")), _num(r.get("ct_justification"))
    r["ct_total"] = None if d is None or j is None else int(d + j)
    comps = []
    for key, f in (("luria_trial1", lambda v: v / 10), ("luria_delayed", lambda v: v / 10),
                   ("schulte_seconds", lambda v: min(max((75 - v) / 51, 0), 1)),
                   ("digit_span_total", lambda v: min(v / 22, 1))):
        v = _num(r.get(key))
        if v is not None:
            comps.append(f(v))
    r["attention_index"] = round(100 * sum(comps) / len(comps), 1) if len(comps) >= 3 else None
    r["low_attention"] = None if r["attention_index"] is None else int(r["attention_index"] < LOW_ATTENTION_CUT)
    r["low_ct"] = None if r["ct_total"] is None else int(r["ct_total"] <= LOW_CT_CUT)
    return r


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


SCHEMA = """
CREATE TABLE students (
    student_id TEXT PRIMARY KEY,
    grade INTEGER CHECK (grade IN (8, 9)), gender TEXT, age INTEGER,
    main_ai_tool TEXT, main_subject TEXT, main_purpose TEXT,
    q4_ai_frequency INTEGER, q7_daily_ai_time INTEGER, q8_ask_ai_first INTEGER,
    q9_copy_directly INTEGER, q10_rewrite_own_words INTEGER, q11_check_correctness INTEGER,
    q12_could_do_without_ai INTEGER,
    screen_time_hours REAL, sleep_hours REAL,
    luria_trial1 INTEGER, luria_trial2 INTEGER, luria_trial3 INTEGER, luria_delayed INTEGER,
    schulte_seconds REAL, digit_span_total INTEGER,
    ct_detection INTEGER, ct_justification INTEGER,
    offloading_score REAL, ai_use_group TEXT, luria_mean REAL, ct_total INTEGER,
    attention_index REAL, low_attention INTEGER, low_ct INTEGER,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)"""


def init_db(force: bool = False):
    """Create the SQLite DB from the CSV on first run (or when force=True)."""
    if DB_PATH.exists() and not force:
        return
    if not CSV_PATH.exists():
        from generate_dataset import generate
        generate().to_csv(CSV_PATH, index=False)
    raw = pd.read_csv(CSV_PATH)
    rows = [derive(r) for r in raw.to_dict("records")]
    df = pd.DataFrame(rows)
    with connect() as con:
        con.execute("DROP TABLE IF EXISTS students")
        con.execute(SCHEMA)
        df.to_sql("students", con, if_exists="append", index=False)


def load_df() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM students ORDER BY student_id", con)


def get(student_id):
    with connect() as con:
        r = con.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
        return dict(r) if r else None


def next_id():
    with connect() as con:
        r = con.execute("SELECT MAX(CAST(SUBSTR(student_id,2) AS INTEGER)) FROM students").fetchone()[0]
    return f"P{(r or 1000) + 1}"


def create(rec: dict) -> str:
    rec = derive(rec)
    rec["student_id"] = next_id()
    cols = ["student_id"] + INPUT_FIELDS + DERIVED
    with connect() as con:
        con.execute(f"INSERT INTO students ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [rec.get(c) for c in cols])
    return rec["student_id"]


def update(student_id, rec: dict):
    rec = derive(rec)
    cols = INPUT_FIELDS + DERIVED
    with connect() as con:
        con.execute(f"UPDATE students SET {','.join(c + '=?' for c in cols)}, updated_at=CURRENT_TIMESTAMP "
                    f"WHERE student_id=?", [rec.get(c) for c in cols] + [student_id])


def delete(student_id):
    with connect() as con:
        con.execute("DELETE FROM students WHERE student_id=?", (student_id,))


def parse_form(form) -> tuple[dict, list]:
    """Validate an HTML form into a record. Returns (record, errors)."""
    rec, errors = {}, []
    for f, opts in CATEGORICAL.items():
        v = form.get(f, "").strip()
        if v and v not in opts:
            errors.append(f"{f}: invalid value")
        rec[f] = v or None
    for f in LIKERT_LABELS:
        v = form.get(f, "").strip()
        if v == "":
            rec[f] = None
        elif v.isdigit() and 1 <= int(v) <= 5:
            rec[f] = int(v)
        else:
            errors.append(f"{f}: must be 1-5")
    for f, (lo, hi, _) in NUMERIC.items():
        v = form.get(f, "").strip()
        if v == "":
            rec[f] = None
            continue
        try:
            x = float(v)
        except ValueError:
            errors.append(f"{f}: must be a number")
            continue
        if not lo <= x <= hi:
            errors.append(f"{f}: must be between {lo} and {hi}")
        rec[f] = x
    if rec.get("grade") is None:
        errors.append("grade: required")
    d, j = rec.get("ct_detection"), rec.get("ct_justification")
    if d is not None and j is not None and j > d:
        errors.append("ct_justification cannot exceed ct_detection (one justification per detected error)")
    return rec, errors
