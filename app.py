"""AI-Use & Cognitive Offloading Analytics - Flask web application.

Pages: Home (KPIs) | Dashboard (filters + charts + statistics) | Records (CRUD) |
Predict (risk of low attention / low critical thinking) | Model (evaluation) | Data (pipeline & quality)
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import hashlib
import io
import json
import math
import threading
from pathlib import Path

import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for

import analysis
import charts
import db
import ml

app = Flask(__name__)
app.secret_key = "pds-mini-project-demo"   # demo only; set FLASK_SECRET_KEY in production
REF_PATH = Path(__file__).parent / "data" / "thesis_reference_n12.csv"

db.init_db()
MODELS, METRICS = ml.load_or_train(db.load_df())


@app.template_filter("fmt")
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return v


@app.template_filter("pval")
def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def apply_filters(df, args):
    f = {k: args.get(k, "") for k in ("grade", "gender", "tool", "group", "purpose")}
    if f["grade"]:
        df = df[df.grade == int(f["grade"])]
    if f["gender"]:
        df = df[df.gender == f["gender"]]
    if f["tool"]:
        df = df[df.main_ai_tool == f["tool"]]
    if f["group"]:
        df = df[df.ai_use_group == f["group"]]
    if f["purpose"]:
        df = df[df.main_purpose == f["purpose"]]
    return df, f


def kpis(df):
    return {
        "n": len(df),
        "users": f"{100 * (df.q4_ai_frequency >= 2).mean():.0f}%" if len(df) else "–",
        "chatgpt": f"{100 * (df.main_ai_tool == 'ChatGPT').mean():.0f}%" if len(df) else "–",
        "offload": df.offloading_score.mean(),
        "attention": df.attention_index.mean(),
        "ct": df.ct_total.mean(),
        "low_att": f"{100 * df.low_attention.mean():.0f}%" if len(df) else "–",
        "low_ct": f"{100 * df.low_ct.mean():.0f}%" if len(df) else "–",
        "copy_rare": f"{100 * (df.q9_copy_directly <= 2).mean():.0f}%" if len(df) else "–",
    }


@app.route("/")
def home():
    df = db.load_df()
    return render_template("home.html", k=kpis(df), m=METRICS)


CHART_CACHE_DIR = Path(__file__).parent / "data" / "chart_cache"
_mem_cache = {}
_cache_lock = threading.Lock()


def _chart_key(full, f):
    """Key = filter params + fingerprint of the data (CRUD edits invalidate)."""
    data_hash = str(int(pd.util.hash_pandas_object(full, index=False).sum()))
    raw = json.dumps(f, sort_keys=True) + "|" + data_hash
    return hashlib.md5(raw.encode()).hexdigest()


def _build_charts(df, ref):
    return {
        "freq": charts.likert_bar(df, "q4_ai_frequency", "Q4 AI use frequency for homework"),
        "time": charts.likert_bar(df, "q7_daily_ai_time", "Q7 Daily time using AI"),
        "tool": charts.cat_barh(df, "main_ai_tool", "Main AI tool (Q3)"),
        "purpose": charts.cat_barh(df, "main_purpose", "Main purpose of AI use (Q6)"),
        "q8": charts.likert_bar(df, "q8_ask_ai_first", "Q8 Ask AI first"),
        "q9": charts.likert_bar(df, "q9_copy_directly", "Q9 Copy AI text directly"),
        "q10": charts.likert_bar(df, "q10_rewrite_own_words", "Q10 Rewrite in own words"),
        "q11": charts.likert_bar(df, "q11_check_correctness", "Q11 Check AI correctness"),
        "hist": charts.offload_hist(df),
        "sc_att": charts.scatter(df, "attention_index", "Attention index"),
        "sc_ct": charts.scatter(df, "ct_total", "Critical thinking /8"),
        "heat": charts.corr_heatmap(df),
        "groups": charts.group_bars(df),
        "luria": charts.luria_curve(df, ref),
    }


def _dashboard_charts(full, df, f, ref):
    """Charts cached in memory + on disk (JSON of base64 PNGs) keyed by filters/data."""
    if len(df) < 5:
        return {}
    key = _chart_key(full, f)
    with _cache_lock:
        if key in _mem_cache:
            return _mem_cache[key]
        path = CHART_CACHE_DIR / f"{key}.json"
        if path.exists():
            try:
                img = json.loads(path.read_text())
                _mem_cache[key] = img
                return img
            except Exception:
                pass
        img = _build_charts(df, ref)
        _mem_cache[key] = img
        try:
            CHART_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(img))
        except OSError:
            pass
        return img


def _warm_dashboard_cache():
    try:
        full = db.load_df()
        f = {k: "" for k in ("grade", "gender", "tool", "group", "purpose")}
        ref = pd.read_csv(REF_PATH) if REF_PATH.exists() else None
        _dashboard_charts(full, full, f, ref)
    except Exception as e:  # never block startup
        print("chart cache warm-up failed:", e)


threading.Thread(target=_warm_dashboard_cache, daemon=True).start()


@app.route("/dashboard")
def dashboard():
    full = db.load_df()
    df, f = apply_filters(full, request.args)
    ref = pd.read_csv(REF_PATH) if REF_PATH.exists() else None
    img = _dashboard_charts(full, df, f, ref)
    return render_template("dashboard.html", f=f, k=kpis(df), img=img, total=len(full),
                           corr=analysis.correlations(df) if len(df) >= 5 else [],
                           crit=analysis.critical_r(len(df)) if len(df) > 3 else None,
                           groups=analysis.group_comparison(df) if len(df) >= 5 else [],
                           profile=analysis.behaviour_profile(df), cats=db.CATEGORICAL)


@app.route("/records")
def records():
    df, f = apply_filters(db.load_df(), request.args)
    q = request.args.get("q", "").strip()
    if q:
        df = df[df.student_id.str.contains(q, case=False)]
    risk = request.args.get("risk", "")
    if risk == "attention":
        df = df[df.low_attention == 1]
    elif risk == "ct":
        df = df[df.low_ct == 1]
    page, per = max(int(request.args.get("page", 1)), 1), 25
    pages = max(math.ceil(len(df) / per), 1)
    rows = df.iloc[(page - 1) * per: page * per].to_dict("records")
    args = {k: v for k, v in request.args.items() if k != "page"}
    return render_template("records.html", rows=rows, f=f, q=q, risk=risk, page=page, pages=pages,
                           n=len(df), args=args, cats=db.CATEGORICAL)


def _full(rec):
    """Ensure every form field key exists so templates can test `is none`."""
    return {**{k: None for k in ["student_id"] + db.INPUT_FIELDS}, **(rec or {})}


def _form_ctx(rec, mode, errors=None):
    return dict(rec=_full(rec), mode=mode, errors=errors or [], cats=db.CATEGORICAL, likert=db.LIKERT_LABELS,
                qtext=db.QUESTION_TEXT, numeric=db.NUMERIC)


@app.route("/records/new", methods=["GET", "POST"])
def record_new():
    if request.method == "POST":
        rec, errors = db.parse_form(request.form)
        if errors:
            return render_template("form.html", **_form_ctx(rec, "new", errors))
        sid = db.create(rec)
        flash(f"Record {sid} created.", "ok")
        return redirect(url_for("record_view", sid=sid))
    return render_template("form.html", **_form_ctx({"grade": 8}, "new"))


@app.route("/records/<sid>")
def record_view(sid):
    rec = db.get(sid)
    if not rec:
        flash(f"Record {sid} not found.", "err")
        return redirect(url_for("records"))
    return render_template("view.html", rec=rec, pred=ml.predict(MODELS, rec), likert=db.LIKERT_LABELS,
                           qtext=db.QUESTION_TEXT)


@app.route("/records/<sid>/edit", methods=["GET", "POST"])
def record_edit(sid):
    rec = db.get(sid)
    if not rec:
        return redirect(url_for("records"))
    if request.method == "POST":
        new, errors = db.parse_form(request.form)
        if errors:
            new["student_id"] = sid
            return render_template("form.html", **_form_ctx(new, "edit", errors))
        db.update(sid, new)
        flash(f"Record {sid} updated.", "ok")
        return redirect(url_for("record_view", sid=sid))
    return render_template("form.html", **_form_ctx(rec, "edit"))


@app.route("/records/<sid>/delete", methods=["POST"])
def record_delete(sid):
    db.delete(sid)
    flash(f"Record {sid} deleted.", "ok")
    return redirect(url_for("records"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    rec, result, errors = {"grade": 9, "gender": "Female", "main_ai_tool": "ChatGPT",
                           "main_purpose": "Explanation", "q4_ai_frequency": 3, "q7_daily_ai_time": 3,
                           "q8_ask_ai_first": 3, "q9_copy_directly": 2, "q10_rewrite_own_words": 4,
                           "q11_check_correctness": 4, "q12_could_do_without_ai": 4,
                           "screen_time_hours": 2.5, "sleep_hours": 8.0}, None, []
    if request.method == "POST":
        rec, errors = db.parse_form(request.form)
        if not errors:
            result = ml.predict(MODELS, rec)
    elif request.args.get("demo") == "high":   # pre-filled high-offloading example for the demo video
        rec.update(main_purpose="Full answer", q4_ai_frequency=5, q7_daily_ai_time=5, q8_ask_ai_first=4,
                   q9_copy_directly=4, q10_rewrite_own_words=2, q11_check_correctness=2,
                   q12_could_do_without_ai=2, screen_time_hours=4.5, sleep_hours=7.0)
        result = ml.predict(MODELS, rec)
    return render_template("predict.html", rec=_full(rec), result=result, errors=errors, cats=db.CATEGORICAL,
                           likert=db.LIKERT_LABELS, qtext=db.QUESTION_TEXT)


@app.route("/model", methods=["GET", "POST"])
def model():
    global MODELS, METRICS
    if request.method == "POST":
        METRICS = ml.train(db.load_df())
        MODELS, _ = ml.load_or_train(db.load_df())
        flash(f"Models retrained on {METRICS['n_records']} records.", "ok")
        return redirect(url_for("model"))
    img = {}
    for t, r in METRICS["targets"].items():
        best = r["models"][r["best_model"]]
        img[t] = {"cm": charts.confusion(best["confusion_matrix"], f"{r['best_model']} (test set)"),
                  "coef": charts.importance_bar(r["lr_coefficients"], "Logistic Regression coefficients "
                                                "(+ = raises risk)", signed=True),
                  "imp": charts.importance_bar(r["rf_importances"], "Random Forest feature importance")}
    return render_template("model.html", m=METRICS, img=img)


@app.route("/data")
def data():
    df = db.load_df()
    raw = pd.read_csv(db.CSV_PATH)
    desc = df[["offloading_score", "attention_index", "luria_trial1", "luria_delayed", "schulte_seconds",
               "digit_span_total", "ct_total", "screen_time_hours", "sleep_hours"]].describe().T.round(2)
    ref = pd.read_csv(REF_PATH) if REF_PATH.exists() else None
    return render_template("data.html", missing=analysis.missing_report(raw), n_raw=len(raw), n_db=len(df),
                           cols=list(raw.columns), desc=desc.reset_index().to_dict("records"),
                           ref=ref.to_dict("records") if ref is not None else [],
                           ref_means=ref[["luria_trial1", "luria_delayed", "ct_total"]].mean().round(2).to_dict()
                           if ref is not None else {}, sim_means=df[["luria_trial1", "luria_delayed",
                                                                      "ct_total"]].mean().round(2).to_dict())


@app.route("/export.csv")
def export_csv():
    df, _ = apply_filters(db.load_df(), request.args)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=ai_offloading_export.csv"})


@app.route("/api/records")
def api_records():
    df, _ = apply_filters(db.load_df(), request.args)
    return jsonify(df.astype(object).where(df.notna(), None).to_dict("records"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    return jsonify(ml.predict(MODELS, request.get_json(force=True) or {}))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
