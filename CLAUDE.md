# Project: Hotel Booking Cancellation Prediction (VoidMain Internship - Task 3)

## Context
- I'm Krishna, B.Tech AI & ML student, AI internship at VoidMain Technologies.
- **Task 3: Machine Learning Application & Model Evaluation** (75 pts, mandatory).
- Official brief: "Develop and compare machine-learning models for a practical problem. Select an appropriate classification or regression problem. Train multiple suitable algorithms, apply appropriate validation and compare models using relevant metrics. Identify overfitting/underfitting where applicable and justify the selected model."
- Standalone submission: own GitHub repo + live deployed app + submit links on the portal.
- Continues my Task 2 (data preprocessing on the same dataset): https://github.com/Krishna141414/hotel-booking-preprocessing - reuse its cleaning/feature logic (clone it to a temp folder and copy what's needed into this repo's `src/`; this repo must run on its own, no dependency on the Task 2 repo at runtime).
- **Continuity, not a redo:** Task 2 already solved cleaning + feature engineering. Reuse that code as-is (copy the needed modules, keep credit to Task 2 in the README) instead of rebuilding or re-exploring the data. Spend the effort on modeling, comparison, overfitting analysis and model selection.
- **Never touch Task 2:** do NOT modify, commit to, or push to the Task 2 repo, and do not change its Render service. Only read/clone it into a temporary folder outside this project, then delete the temp clone.
- **No video files or video scripts.** Just create an empty `screenshots/` folder (with `.gitkeep`) for me to add screenshots later.
- Environment: Windows, VS Code, Python 3.14 (same as Task 2). Project is in OneDrive Desktop - keep `.venv` out of git.

## Problem
- **Binary classification:** predict whether a hotel booking will be canceled (`is_canceled`).
- Business framing: help hotels anticipate cancellations to manage overbooking and revenue.
- Dataset: Hotel Booking Demand (Antonio, Almeida & Nunes, 2019). Download automatically from
  `https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv`
  (fallback: `master` instead of `main`), save to `data/raw/hotel_bookings.csv`, verify 119,390 x 32. Treat `"NULL"` as missing.
- Apply the same cleaning as Task 2 (drop exact duplicates, 0-guest rows, negative ADR, cap extreme ADR, etc.).
- **Leakage:** drop `reservation_status` and `reservation_status_date` (they reveal the outcome). Also review `booking_changes`, `days_in_waiting_list`, `assigned_room_type` for post-booking leakage and decide with a documented reason.
- Class balance after cleaning is ~27.5% canceled - handle imbalance appropriately (stratified splits, class weights where relevant) and don't rely on accuracy alone.

## Required ML work (every brief requirement must be clearly visible)
1. **Validation setup:** stratified train/test split (hold out test set, touched only once at the end) + stratified 5-fold cross-validation on train. Fixed random_state.
   - **Extra realism check - temporal validation:** also train on earlier arrivals and test on the latest period (e.g. by arrival date) to see if performance holds over time (data drift). Report it next to the random split and discuss the difference.
2. **Baseline:** DummyClassifier (most frequent) so every model has a reference point.
3. **Multiple algorithms** (sklearn, each inside a Pipeline with preprocessing):
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - HistGradientBoosting (or Gradient Boosting)
   - K-Nearest Neighbors (on a stratified sample if too slow - document it)
   - Optional: XGBoost/LightGBM only if they install cleanly on Python 3.14; otherwise skip and note why
4. **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix; CV mean +/- std per model; training/prediction time.
5. **Hyperparameter tuning:** RandomizedSearchCV on the top 2 models (reasonable budget).
6. **Overfitting/underfitting analysis:** train vs CV vs test scores table, learning curves for the top models, validation curve (e.g. tree depth vs score). Clearly label which models over/underfit and why.
7. **Model selection + justification:** choose the final model using CV metrics + generalization gap + speed + interpretability + deployment size. Write the reasoning.
8. **Threshold tuning:** pick a decision threshold based on the business trade-off (e.g. maximize F1 or meet a recall target) and show its effect.
9. **Explainability:** permutation importance (and feature importance for tree models) with plain-English insights.
10. **Final evaluation:** final model on the untouched test set + classification report + ROC/PR curves + confusion matrix.
11. **Decisions log:** every decision in `docs/decisions.md` (what, why, alternative considered).

## Project structure
```
data/raw/              # downloaded CSV (commit it, ~16 MB)
data/processed/        # gitignored
notebooks/modeling.ipynb   # full walkthrough, run top to bottom with outputs saved
src/
  config.py            # paths, URLs, columns, random_state, thresholds
  download_data.py
  data.py              # load + clean (adapted from Task 2)
  features.py          # feature engineering (adapted from Task 2)
  models.py            # model zoo + pipelines
  train.py             # CV, tuning, selection, final fit
  evaluate.py          # metrics, curves, overfitting analysis
  visualize.py         # shared colorful plot style
models/                # final_model.joblib + metrics.json (keep small for deploy)
reports/figures/       # saved plots
reports/model_comparison.csv
app/                   # Flask web app (app.py, templates/, static/)
tests/                 # pytest
screenshots/           # EMPTY with .gitkeep
docs/decisions.md
main.py                # one command: download -> clean -> train/compare -> evaluate -> save
README.md
requirements.txt / requirements-dev.txt
render.yaml, gunicorn.conf.py
```

## Web app (Flask, deployed on Render free tier)
- Custom HTML/CSS/JS, **no Streamlit**. Professional and **a bit colorful but clean** - modern gradients/accents, good typography, cards, smooth charts (Chart.js or Plotly.js), light/dark mode. Aesthetic, not flashy.
- Pages:
  1. **Overview** - problem, dataset, final model, headline metrics
  2. **Model comparison** - interactive table + charts of all models (CV and test metrics, timings)
  3. **Overfitting analysis** - train vs CV vs test, learning curves
  4. **Explainability** - feature importance + insights
  5. **Predict** - form for booking details -> cancellation probability with a clear risk gauge (+ "try a random sample booking" button)
- **Color palette (use consistently in the web app AND all matplotlib/seaborn figures):**
  - Base: deep midnight navy `#0B1020` (dark mode bg) / soft off-white `#F7F8FC` (light mode bg), surfaces `#141A33` / `#FFFFFF`
  - Primary: electric blue `#4F7CFF`
  - Accent 1: neon magenta-pink `#FF3D81` (highlights, "canceled" class, risk)
  - Accent 2: marigold `#FFB020` (warnings, selected model badge, key numbers)
  - Accent 3: mint teal `#22D3B6` (success, "not canceled" class, good metrics)
  - Use subtle gradients (blue -> pink) for hero/headers, soft neon glow on key cards/buttons, rounded cards, plenty of whitespace. Colorful but tasteful - never noisy. Ensure good contrast/readability in both modes.
- **Charts & visuals are a core part of the app** (interactive where possible, Chart.js or Plotly.js, same palette):
  - Model comparison: grouped bar chart of metrics per model + radar chart across metrics + sortable table
  - ROC curves and Precision-Recall curves of all models on one chart
  - Confusion matrix heatmaps (final model, plus toggle for other models)
  - Train vs CV vs test score chart (overfitting gaps) + learning curves + validation curve (tree depth)
  - **Decision tree visualization:** show the top levels (depth 3-4) of the Decision Tree as a readable, collapsible/zoomable interactive tree (e.g. D3 tree from exported JSON) + a static plot_tree image in reports/figures
  - Feature importance / permutation importance bar charts
  - Threshold slider on the final model: shows how precision/recall/F1 change live
  - Predict page: animated risk gauge with the probability
- **Wow features (impressive but stable - skip any that risk breaking the app or the memory limit, and note it):**
  - **3D booking space:** interactive Plotly 3D scatter of ~2,000 sampled bookings projected with PCA (3 components), colored canceled vs not, rotatable/zoomable
  - **Model switcher:** a toggle/pill selector at the top of analysis pages; picking a model updates all its charts (curves, confusion matrix, metrics) with smooth transitions
  - **Head-to-head mode:** pick any 2 models and compare them side by side
  - **What-if simulator** on Predict: sliders/toggles (lead time, ADR, deposit type, special requests, etc.) update the cancellation probability live
  - **"Why this prediction":** per-feature contribution bars for the current prediction (from the model's own interpretable signal, no heavy extra libraries)
  - **Business impact calculator:** user sets cost of an empty room vs cost of overbooking -> best threshold + estimated savings
  - **"Falling bookings" tree animation** (inspired by R2D3's "A Visual Introduction to Machine Learning"): sample bookings drop as colored dots through the top splits of the Decision Tree and land in canceled / not-canceled buckets; replay button. Precompute paths as JSON, animate with D3/canvas.
  - **Decision boundary playground** (TensorFlow-Playground style): pick 2 features (e.g. lead_time vs adr) and a model -> shows its decision regions over a precomputed grid with sample points. Visually shows overfitting (jagged deep tree) vs underfitting (straight logistic line).
  - **Training replay:** animated chart of Gradient Boosting's validation score improving stage by stage (from staged_predict_proba), with play/pause.
  - **Kaggle-style leaderboard:** models ranked with medals, score bars, and a "why it won" card for the selected model.
  - Polished micro-interactions: animated count-up numbers, fade/slide-in on scroll, skeleton loaders, hover states, fully responsive on mobile
- Must fit Render free (512 MB): precompute all metrics/chart data into small cached JSON; load only the final model; keep model file small (compress / limit tree size). Health check at `/healthz`.

## Build priority (so one run always ends with a working project)
1. All "Required ML work" + figures + decisions log + tests (must)
2. Core web app pages + charts + Predict page + deploy readiness (must)
3. Wow features, in this order: leaderboard, model switcher, threshold slider, what-if simulator, falling-bookings tree, decision boundary playground, 3D booking space, training replay, head-to-head, business calculator
Build in this order; if something in level 3 gets too complex or heavy, skip it and list it in the final summary.

## Engineering standards (professional, NOT vibe-coded)
- Type hints + Google-style docstrings everywhere; small functions; ruff format + check clean.
- No hardcoded values outside `src/config.py`; `logging` with clean before/after output.
- Reproducible: fixed random_state, pinned versions, `python main.py` runs everything.
- pytest tests for data, features, models, app routes; all must pass.
- No leakage: preprocessing fit inside pipelines on training folds only.
- Consistent, colorful, well-labeled figures saved to `reports/figures/`.
- Clean git history: one commit per logical step with clear messages (`feat:`, `docs:`, `test:`...).
- Professional README: overview, live link, problem, dataset credit, pipeline diagram, model comparison table, overfitting findings, selected model + justification, key figures, how to run, structure, tech stack, link to Task 2 repo.

## How to work
- Do EVERYTHING in one go, without stopping for approval between steps.
- If a decision could reasonably go either way, choose the safer option and log it in docs/decisions.md.
- **Self-review before finishing:** go through every item in "Required ML work", the web app list and the engineering standards like a strict reviewer; check every page and chart actually renders with real data (no placeholders, no console errors), light + dark mode, mobile width; fix anything missing or broken.
- At the end: run main.py, pytest, ruff, start the app locally to confirm it works, then give ONE summary (model comparison table, selected model + why, overfitting findings, test metrics, app pages, how to run, things to double-check). Then stop.
- Explain casually and briefly; I prefer direct, practical answers.
