"""ML pipeline: predict LOW ATTENTION and LOW CRITICAL THINKING from AI-use / offloading features.

Only questionnaire + lifestyle features are used as inputs (no cognitive test scores),
so the model answers the thesis question: "can AI-offloading behaviour flag students
at risk of weaker attention / critical thinking before they are tested?"
For each target: Logistic Regression (interpretable) vs Random Forest, stratified 75/25 split
+ 5-fold stratified CV; best model by F1 of the at-risk class is saved.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = Path(__file__).parent / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
SEED = 42

CAT_FEATURES = ["gender", "main_ai_tool", "main_purpose"]
NUM_FEATURES = ["grade", "q4_ai_frequency", "q7_daily_ai_time", "q8_ask_ai_first", "q9_copy_directly",
                "q10_rewrite_own_words", "q11_check_correctness", "q12_could_do_without_ai",
                "offloading_score", "screen_time_hours", "sleep_hours"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGETS = {"low_attention": "Low attention (attention index < 55)",
           "low_ct": "Low critical thinking (CT total <= 4/8)"}


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ])


def _X(df):
    X = df[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _model_path(target):
    return MODEL_DIR / f"{target}_classifier.joblib"


def train(df: pd.DataFrame) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    out = {"n_records": int(len(df)), "features": FEATURES, "targets": {}}
    for target, label in TARGETS.items():
        d = df.dropna(subset=[target])
        X, y = _X(d), d[target].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        cands = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
            "Random Forest": RandomForestClassifier(n_estimators=400, random_state=SEED, class_weight="balanced",
                                                    min_samples_leaf=3, max_depth=6),
        }
        res, best, best_f1 = {}, None, -1
        for name, est in cands.items():
            pipe = Pipeline([("prep", preprocessor()), ("model", est)]).fit(Xtr, ytr)
            p, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
            m = {
                "accuracy": accuracy_score(yte, p),
                "precision": precision_score(yte, p, zero_division=0),
                "recall": recall_score(yte, p),
                "f1": f1_score(yte, p),
                "f1_macro": f1_score(yte, p, average="macro"),
                "roc_auc": roc_auc_score(yte, proba),
                "cv5_f1": float(cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()),
                "cv5_roc_auc": float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()),
            }
            res[name] = {k: round(float(v), 4) for k, v in m.items()}
            res[name]["confusion_matrix"] = confusion_matrix(yte, p, labels=[0, 1]).tolist()
            if m["f1"] > best_f1:
                best, best_f1, best_name = pipe, m["f1"], name
        # baseline: always predict the majority class
        res["Majority baseline"] = {"accuracy": round(float((yte == y.mode()[0]).mean()), 4), "f1": 0.0}

        # interpretability: LR coefficients (direction) + RF importances (strength), fitted on all data
        lr = Pipeline([("prep", preprocessor()), ("model", cands["Logistic Regression"])]).fit(X, y)
        names = [n.split("__", 1)[1] for n in lr.named_steps["prep"].get_feature_names_out()]
        coefs = sorted(zip(names, lr.named_steps["model"].coef_[0]), key=lambda t: -abs(t[1]))[:10]
        rf = Pipeline([("prep", preprocessor()), ("model", cands["Random Forest"])]).fit(X, y)
        imps = sorted(zip(names, rf.named_steps["model"].feature_importances_), key=lambda t: -t[1])[:10]

        joblib.dump(best, _model_path(target))
        out["targets"][target] = {
            "label": label, "positive_rate": round(float(y.mean()), 3),
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "models": res, "best_model": best_name,
            "lr_coefficients": [(n, round(float(v), 3)) for n, v in coefs],
            "rf_importances": [(n, round(float(v), 4)) for n, v in imps],
        }
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    return out


def load_or_train(df):
    if METRICS_PATH.exists() and all(_model_path(t).exists() for t in TARGETS):
        models = {t: joblib.load(_model_path(t)) for t in TARGETS}
        return models, json.loads(METRICS_PATH.read_text())
    m = train(df)
    return {t: joblib.load(_model_path(t)) for t in TARGETS}, m


def predict(models: dict, record: dict) -> dict:
    from db import derive
    rec = derive(record)
    X = _X(pd.DataFrame([{f: rec.get(f) for f in FEATURES}]))
    out = {"offloading_score": rec.get("offloading_score")}
    for t, model in models.items():
        p = float(model.predict_proba(X)[0, 1])
        out[t] = {"probability": round(p, 3), "risk": "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"}
    return out


if __name__ == "__main__":
    from db import init_db, load_df
    init_db()
    m = train(load_df())
    for t, r in m["targets"].items():
        print(t, "best:", r["best_model"], {k: (v.get("f1"), v.get("roc_auc"), v.get("cv5_f1")) for k, v in r["models"].items()})
