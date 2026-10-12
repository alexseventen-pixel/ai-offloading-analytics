# Student AI-Use & Cognitive Offloading Analytics

Programming for Data Science, Master's mini project. Riga Nordic University, 2026. Author: Oleg Mishchenko.

This is a Flask + SQLite + scikit-learn web app built on the Master's thesis *"Use of AI Tools and Cognitive Offloading among
8th and 9th Grade Students: Associations with Attention and Critical Thinking in a Latvian International School"*.
It turns the thesis instruments into a working analytics tool:

* the 12-item AI-use questionnaire, and the **AI Offloading Composite Score** built from Q4, Q8, Q9, 6−Q10, 6−Q11 and 6−Q12 on a 1–5 scale
* **Luria's 10 Words Test**, **Schulte Tables** and **Digit Span**, combined into an **Attention index** from 0 to 100
* the **Critical Thinking "Survival Story"** fact-checking task (detection 0–4 plus justification 0–4, so 0–8)

> **The data are synthetic.** The thesis tested 12 students. Its abstract says 22, and the two figures conflict.
> This project simulates 180 students using the thesis instruments and distributions, so a sample of that size can
> test the hypothesis properly. The real n = 12 thesis results (Tables 4.4 and 4.6) are in
> `data/thesis_reference_n12.csv` and are shown in the app next to the simulated data.

## Run

```bash
cd ai-offloading-analytics
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                        # open http://127.0.0.1:5000
```

Optional steps:

```bash
python generate_dataset.py                       # rebuild data/ai_offloading_students.csv (seed 2026)
python -c "import db; db.init_db(force=True)"    # rebuild data/ai_offloading.db from the CSV
python ml.py                                     # retrain models and rewrite models/metrics.json
python -m unittest discover -s tests -v          # 8 unit and integration tests
```

If the DB or the models are missing, they are created automatically on the first `python app.py`.

## App modules

| Page | What it does |
|---|---|
| Home | KPIs: number of records, % AI users, mean offloading, attention and CT, % flagged at risk |
| Dashboard | Filters (grade, gender, tool, purpose, AI-use group). Charts for Q3–Q11, an offloading histogram, scatter plots with regression lines, a correlation heatmap and the Luria learning curve set against the thesis n = 12. Tables of Pearson/Spearman r with p-values and Low vs Higher AI-use groups (Welch t-test, Cohen's d), matching thesis Tables 4.3, 4.7 and 4.8. CSV export |
| Records (CRUD) | List, search, filter and paginate records. Create, view, edit and delete them. Input is validated, and derived scores and flags are recalculated on save |
| Risk Prediction | Questionnaire answers go in and the app returns the probability of **low attention** (index < 55) and **low critical thinking** (≤ 4/8), with recommendations from thesis Ch. 5.3. `?demo=high` loads an example |
| Model Evaluation | Accuracy, precision, recall, F1, ROC-AUC and 5-fold CV for Logistic Regression, Random Forest and a majority baseline. Confusion matrices, LR coefficients, RF importances and a Retrain button |
| Data & Pipeline | Pipeline description, missing-values report, descriptive statistics and the thesis reference table. JSON API at `/api/records` and `POST /api/predict` |

## Structure

```
app.py                 Flask routes
db.py                  SQLite schema, CRUD, validation, derived thesis indicators
analysis.py            correlations, critical r, group comparison, behaviour profile, missing report
ml.py                  preprocessing pipeline, classifiers, evaluation, prediction
charts.py              matplotlib charts rendered as base64 PNG
generate_dataset.py    synthetic thesis-aligned data generator
data/                  ai_offloading_students.csv, ai_offloading.db, thesis_reference_n12.csv
models/                low_attention_classifier.joblib, low_ct_classifier.joblib, metrics.json
templates/, static/    UI
tests/test_app.py      unit and integration tests
docs/screenshots/      screenshots for the report
REPORT_DRAFT.md        report draft covering all 25 guideline sections
PRESENTATION_15_SLIDES.md
```

## Submission checklist (from the guidelines)

ZIP containing the code, data/DB and models, the report as DOCX/PDF and the PPT. Demo video on Google Drive, shared as
*Anyone with the link can view*. Put the link in the Moodle online text.
