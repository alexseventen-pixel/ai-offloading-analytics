# Presentation: Student AI-Use & Cognitive Offloading Analytics
*15 slides, following the guideline structure. Each slide has its on-screen content and speaker notes (about 1 minute per slide).*

---

## Slide 1 · Title
**Student AI-Use & Cognitive Offloading Analytics**
A data-science system for attention and critical-thinking risk screening, grades 8–9
Oleg Mishchenko · Programming for Data Science · Riga Nordic University · 2026
Supervisor: Amit Joshi [verify]
*Visual: app home page screenshot (`docs/screenshots/home.png`)*

> Notes: The project turns my Master's thesis on AI tools and cognitive offloading into a working analytics application.

---

## Slide 2 · Problem Statement
* AI tools such as ChatGPT and Gemini are now part of everyday homework: 100% of students in the thesis sample use them
* Literature: AI reliance ↔ lower critical thinking (r = −0.35 to −0.68); more than 2 h/day of screen time gives 51% higher risk of concentration problems
* Latvia: 42% of students are distracted by devices in class (OECD average 30%)
* Schools have **no student-level data or tools**, and the thesis sample (n = 12) was too small to test the hypothesis

> Notes: The open question is whether AI is a support or a substitute for thinking, and nobody measures this at school level.

---

## Slide 3 · Objectives
* **Main:** build a web system that operationalises the thesis methodology and predicts at-risk students
* Model the instruments: 12-item questionnaire, Luria, Schulte, Digit Span, Critical Thinking task
* Automatic **AI Offloading Composite** (Q4, Q8, Q9, reversed Q10–Q12)
* Statistics that mirror thesis Tables 4.3, 4.7 and 4.8
* ML: predict **low attention** and **low critical thinking** from questionnaire answers only
* CRUD, dashboard, API. Target: F1 ≥ 0.60, AUC ≥ 0.75 ✔

---

## Slide 4 · Literature Review
| Study | Finding |
|---|---|
| Gerlich 2025 (n = 655) | AI use ↔ offloading +0.72; ↔ CT −0.68 |
| Tian & Zhang 2025 (n = 580) | AI dependence ↔ CT −0.35; fatigue mediates |
| Risko & Gilbert 2016 | Offloading: performance ↑, retention ↓ |
| Leonhardt et al. 2025 (33 studies) | > 2 h screen time → +51% concentration risk |
| Sina et al. 2024 (I.Family, 8,673) | Smartphone hours → impulsivity ↑ |
| OECD PISA 2022 | Latvia 42% digital distraction |

**Gap:** no studies on Baltic lower-secondary students, and no operational tools for teachers

---

## Slide 5 · Methodology
1. Extract instruments and hypothesis from the thesis
2. Generate synthetic, thesis-calibrated data (N = 180, seed 2026, about 3% missing)
3. Load into SQLite and derive the offloading score, attention index, CT total and risk flags
4. Statistics: Pearson/Spearman, Welch t-test, Cohen's d
5. ML: Logistic Regression vs Random Forest, stratified 75/25 split plus 5-fold CV, selection by F1 of the at-risk class
6. Flask web app, tests, documentation
*Development: iterative, 7 work packages*

---

## Slide 6 · Architecture Diagram
*Visual: system architecture diagram (REPORT §9.2) and ML pipeline (§9.5)*
Browser → Flask routes → {db.py CRUD and derive · analysis.py SciPy · ml.py scikit-learn · charts.py matplotlib} → SQLite DB, CSV, joblib models

> Notes: Three tiers. Teachers add records, the indicators are recomputed, and the models can be retrained from the UI.

---

## Slide 7 · Technology Stack
| Layer | Tools |
|---|---|
| Frontend | HTML5, Jinja2, CSS3 |
| Backend | Python 3.13, Flask 3.1, REST API |
| Database | SQLite 3 |
| Data science | pandas, NumPy, SciPy, scikit-learn, matplotlib, joblib |
| Quality | unittest (8 tests), fixed seeds |
| Deployment | local; gunicorn, Docker or Render ready |

---

## Slide 8 · Dataset Description
* **180 synthetic students** (grade 8: 97 / grade 9: 83; 104 male / 76 female), 24 raw and 7 derived attributes
* Questionnaire Q3–Q12, screen time, sleep · Luria T1–T3 and delayed · Schulte (s) · Digit Span · CT detection and justification
* Calibrated to the thesis: ChatGPT 61% (thesis 67%), Luria T1 6.8 (6.8), CT 5.3/8 (5.6), offloading 2.21 (2.00)
* Real thesis data (n = 12) kept as a reference table
* Cleaning: reverse coding, partial composites, range validation, median or mode imputation inside the pipeline

> Notes: The data are synthetic because n = 12 is too small for ML and data on minors cannot be published.

---

## Slide 9 · System Modules
*Visual: 4 screenshots (dashboard, records, predict_result, model)*
* **Dashboard:** 5 filters, 14 charts, correlation and group tables, CSV export
* **Records:** CRUD with validation, search, risk badges
* **Risk Prediction:** answers → probabilities → recommendations ("think first, then verify with AI")
* **Model Evaluation:** metrics, confusion matrices, coefficients, retrain
* **Data & Pipeline:** missing values, statistics, thesis reference · JSON API

---

## Slide 10 · Gantt Chart
*Visual: Gantt chart from REPORT §15*
WP1 Requirements (wk 1) → WP2 Dataset (wk 2) → WP3 Preprocessing (wk 3) → WP4 Models (wk 4–5) ∥ WP5 Frontend (wk 4–5) → WP6 Testing (wk 6) → WP7 Docs, video, ZIP (wk 7)
Milestones: data ready (wk 3) · app feature-complete (wk 5) · submission (wk 7)

---

## Slide 11 · SWOT Analysis
| Strengths | Weaknesses |
|---|---|
| Based on thesis and validated tests · interpretable · screening from questionnaire only · lightweight, tested | Synthetic data · self-report bias · non-clinical cut-offs · no login yet |
| **Opportunities** | **Threats** |
| Real follow-up study with 40–50+ students · longitudinal tracking · transfer to HR and teams · EU AI-literacy policy | GDPR and minors · stigmatising labels · AI tools change fast · confounding by ability |

---

## Slide 12 · Results & Analysis
* Offloading ↔ attention index **r = −0.58**; ↔ critical thinking **r = −0.53** (p < 0.001; critical r = 0.146, vs 0.576 at n = 12)
* Higher vs Low AI-use group: attention 59.5 vs 70.8 (d = −0.90), CT 4.6 vs 6.1 (d = −0.74)
* **Low attention model (LR):** F1 0.76 · recall 1.00 · AUC 0.96 (CV F1 0.66)
* **Low CT model (RF):** F1 0.65 · AUC 0.75 (CV F1 0.68), both above the majority baseline
* Top predictors: offloading score, Q12 independence, Q11 checking, screen time, sleep
*Visual: two scatter plots and the confusion matrix*

> Notes: How students use AI matters as much as how often. The thesis's own n = 12 result is within sampling noise.

---

## Slide 13 · Future Scope
* Real, consented data from the next cohort (40–50+ students) and longitudinal design
* SHAP explanations, gradient boosting, mediation analysis (AI → offloading → CT)
* Digital Schulte and Digit Span tests with reaction times
* Authentication, RBAC, PostgreSQL, Docker and cloud deployment, Moodle integration
* Transfer to organisations: AI-reliance screening of teams

---

## Slide 14 · Conclusion
* The thesis methodology now runs as a reproducible pipeline and web tool: data model, database, CRUD, statistics, ML and dashboard
* At an adequate sample size, the hypothesised negative link between offloading and cognition is detectable
* Questionnaire-only screening reaches useful accuracy, as an advisory aid and not a diagnosis
* Ready for the real follow-up study recommended by the thesis
**Demo:** live app or recorded video (Google Drive link, "Anyone with the link")

---

## Slide 15 · References
1. Gerlich, M. (2025). AI tools in society: impacts on cognitive offloading and the future of critical thinking. *Societies, 15*(1), 6.
2. Tian, X., & Zhang, J. (2025). AI dependence, cognitive fatigue, and critical thinking.
3. Risko, E. F., & Gilbert, S. J. (2016). Cognitive offloading. *Trends in Cognitive Sciences, 20*(9).
4. Leonhardt, M., et al. (2025). Screen time and concentration difficulties.
5. Sina, E., et al. (2024). Digital media use and cognition in European children (I.Family).
6. OECD (2023). *PISA 2022 Results, Vol. I.*
7. Zhai, C., Wibowo, S., & Li, L. D. (2024). Over-reliance on AI dialogue systems. *Smart Learning Environments, 11*.
8. Mishchenko, O. (2026). Master's thesis, Riga Nordic University.
9. Pedregosa, F., et al. (2011). Scikit-learn. *JMLR, 12*. Flask, pandas and SciPy documentation.

---

### Demo video script (3–5 minutes)
1. Home: KPIs and the synthetic-data note
2. Dashboard: apply Grade 9 + Higher AI use, then scroll through the charts, correlation table and t-test table
3. Records: filter "Low critical thinking", open a record, edit a score, show the recomputed CT total and flags, create a new record, delete it
4. Risk Prediction: enter a reflective profile (Low risk), then open `?demo=high` (High risk) and read the recommendations
5. Model Evaluation: metrics, confusion matrix and coefficients, then click Retrain
6. Data & Pipeline: missing values, statistics, thesis n = 12 table, CSV export
