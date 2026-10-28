# Hotel Booking Cancellation Prediction: Model Comparison, Tuning & Selection

Six machine-learning models trained, cross-validated, tuned, stress-tested and explained for
predicting whether a hotel booking will be canceled, with a live web app to explore the
results and score new bookings.

![Python](https://img.shields.io/badge/python-3.14-4F7CFF)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-FF3D81)
![Tests](https://img.shields.io/badge/tests-136%20passing-22D3B6)
![Lint](https://img.shields.io/badge/lint-ruff-FFB020)

> **Live web app:** [hotel-cancellation-prediction.onrender.com](https://hotel-cancellation-prediction.onrender.com) _(free tier: the first visit after idle can take ~30-60 s)_
> &nbsp;|&nbsp; **Decisions log:** [`docs/decisions.md`](docs/decisions.md)
> &nbsp;|&nbsp; **Walkthrough notebook:** [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb)
> &nbsp;|&nbsp; **Task 2 (data preprocessing):** [hotel-booking-preprocessing](https://github.com/Krishna141414/hotel-booking-preprocessing)

VoidMain Technologies AI Internship - **Task 3: Machine Learning Application & Model Evaluation**.

---

## Overview

| Brief requirement | Where |
|---|---|
| A practical classification problem | Predict `is_canceled` at booking time (section [Problem](#problem)) |
| Multiple suitable algorithms | Dummy baseline + Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, KNN |
| Appropriate validation | Stratified hold-out test set (used once) + stratified 5-fold CV + temporal check |
| Compare with relevant metrics | PR-AUC (primary), ROC-AUC, F1, precision, recall, accuracy, confusion matrix, fold std, timings, size |
| Over/underfitting | Train vs CV vs test, learning curves, depth validation curve, boosting replay, decision boundaries |
| Justify the selected model | CV-only selection rule + generalisation gap + size + speed + interpretability ([section](#selected-model)) |

Everything runs with one command: `python main.py`.

## Problem

Hotels lose revenue when a canceled room stays empty, and lose goodwill when they overbook
and have to relocate a guest. Predicting which bookings will cancel lets a revenue manager
overbook *just enough*. The target is `is_canceled` (1 = canceled), predicted from what is
known **when the booking is made**. About **27.5%** of bookings are canceled, so
accuracy alone would mislead (always guessing "not canceled" is already 72.5% accurate).
Models are ranked on **PR-AUC**, the precision-recall trade-off for the canceled class.

## Dataset

**Hotel booking demand**: Nuno Antonio, Ana de Almeida, Luis Nunes (2019).
*Hotel booking demand datasets.* Data in Brief, 22, 41-49.
[doi.org/10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126)

- Real bookings from a city hotel and a resort hotel in Portugal (arrivals July 2015 – August 2017).
- 119,390 rows x 32 columns, downloaded automatically from the public
  [TidyTuesday mirror](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11)
  (`"NULL"` read as missing, shape verified) and committed in `data/raw/`.
- **Cleaning and feature engineering are reused unchanged from Task 2**
  ([hotel-booking-preprocessing](https://github.com/Krishna141414/hotel-booking-preprocessing)):
  119,390 -> **87,208** rows (exact duplicates, zero-guest rows, negative ADR and impossible
  counts removed; ADR capped; structural missing values filled) and 10 engineered features
  -> **33 model inputs**.
- **Leakage:** `reservation_status(_date)` (the outcome itself), `assigned_room_type` (set at
  check-in) and `booking_changes` (counted up to the outcome date) are excluded. An ablation
  confirms that adding them back would inflate CV PR-AUC by +0.019
  ([decisions 3](docs/decisions.md#3-leakage-review)).

## Pipeline

```mermaid
flowchart LR
    A[Raw CSV<br/>119,390 x 32] --> B[Task 2 cleaning<br/>+ features]
    B --> C[Leakage review<br/>33 inputs]
    C --> D[Stratified split<br/>80 / 20]
    D --> E[5-fold CV<br/>6 models in pipelines]
    E --> F[RandomizedSearchCV<br/>top 2, one-SE rule]
    F --> G[Select on CV:<br/>score, gap, size, speed]
    G --> H[Threshold on<br/>out-of-fold preds]
    H --> I[Test set<br/>used once]
    I --> J[Explain + temporal<br/>drift check]
    J --> K[final_model.joblib<br/>+ JSON cache]
    K --> L[Flask web app]
```

## Model comparison

5-fold stratified CV on the training set (mean ± fold std) and the held-out test set, both
at the default 0.5 threshold for F1. Fit time is on the full training set. KNN is trained on a stratified 20,000-row sample (it compares every prediction with every stored row). Full table: [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Rank | Model | CV PR-AUC | CV ROC-AUC | CV F1 | Train PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 | Gap | Diagnosis | Fit (s) | Size (MB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | **HistGradientBoosting (tuned)** ★ | 0.779 ± 0.005 | 0.904 | 0.708 | 0.840 | 0.786 | 0.907 | 0.715 | 0.060 | Mild overfitting | 4.9 | 0.71 |
| 2 | Random Forest (tuned) | 0.779 ± 0.005 | 0.903 | 0.706 | 0.899 | 0.785 | 0.907 | 0.716 | 0.120 | Strong overfitting | 44.4 | 29.44 |
| 3 | HistGradientBoosting | 0.778 ± 0.004 | 0.903 | 0.707 | 0.859 | 0.787 | 0.907 | 0.715 | 0.081 | Mild overfitting | 3.1 | 0.45 |
| 4 | Random Forest | 0.773 ± 0.005 | 0.903 | 0.715 | 1.000 | 0.776 | 0.906 | 0.725 | 0.227 | Strong overfitting | 14.7 | 69.12 |
| 5 | Logistic Regression | 0.683 ± 0.014 | 0.858 | 0.654 | 0.683 | 0.680 | 0.856 | 0.654 | 0.000 | Underfitting | 1.0 | 0.01 |
| 6 | K-Nearest Neighbors | 0.668 ± 0.012 | 0.837 | 0.582 | 1.000 | 0.675 | 0.839 | 0.581 | 0.332 | Strong overfitting | 0.2 | 1.40 |
| 7 | Decision Tree | 0.479 ± 0.005 | 0.730 | 0.607 | 1.000 | 0.477 | 0.727 | 0.602 | 0.521 | Strong overfitting | 1.2 | 0.38 |
| – | Dummy (most frequent) | 0.275 ± 0.000 | 0.500 | 0.000 | 0.275 | 0.275 | 0.500 | 0.000 | 0.000 | Baseline | 0.4 | 0.01 |

![CV comparison](reports/figures/02_cv_model_comparison.png)

| ROC curves (test) | Precision-recall curves (test) |
|---|---|
| ![ROC](reports/figures/03_roc_curves.png) | ![PR](reports/figures/04_pr_curves.png) |

## Overfitting & underfitting

Train scores are the mean over the CV training folds, so "train vs CV" compares the very same
fits. Rule: gap > 0.05 PR-AUC = overfitting (> 0.10 strong); small gap but > 0.05 below the
best model = underfitting.

- **Decision Tree (no depth limit): strong overfitting.** Train 1.000 vs CV 0.479: it memorises individual bookings. The validation curve peaks at **depth 12**; deeper trees keep improving on train while CV falls.
- **KNN: strong overfitting.** Distance weighting scores every training row perfectly (train 1.000); CV 0.668.
- **Random Forest: large but benign gap.** Fully grown trees memorise (train 1.000), yet averaging still gives a top CV score (0.773). Tuning (deeper leaves) shrinks the gap to 0.120.
- **Logistic Regression: underfitting.** Train = CV (0.683 / 0.683), about 0.10 below the boosted model: a straight boundary cannot capture the interactions. Its learning curve is flat, so more data would not help.
- **HistGradientBoosting: best balance.** HistGradientBoosting (tuned) has a gap of 0.060 (mild overfitting) with the best CV score. Tuning uses a **one-standard-error rule**: among settings within one fold-std of the best CV score it picks the least-overfitting one, so tuning cannot buy noise-level gains with memorisation.
- The **decision-boundary playground** in the app shows the same thing visually: a deep tree on 2 features draws jagged islands (train PR-AUC ≈ 1.0, test far lower), while logistic regression draws one straight line.

![Train vs CV vs test](reports/figures/06_overfitting_gaps.png)

| Learning curves | Decision Tree depth (validation curve) |
|---|---|
| ![Learning curves](reports/figures/07_learning_curves.png) | ![Validation curve](reports/figures/08_validation_curve_depth.png) |

## Selected model

### ★ HistGradientBoosting (tuned)

Chosen by a rule that only looks at cross-validation (`src/train.py::select_final_model`):

1. **CV score:** best mean CV PR-AUC (0.7794 ± 0.0051). Models within 0.005 count as tied: HistGradientBoosting (tuned) (0.7794), Random Forest (tuned) (0.7786), HistGradientBoosting (0.7776).
2. **Generalisation gap:** the smallest train-CV gap among the tied models (0.060) vs 0.120 for Random Forest (tuned).
3. **Deployment size and speed:** 0.71 MB and 7 ms per 1,000 predictions, vs 29.4 MB / 35 ms for Random Forest (tuned). That matters on a 512 MB free instance.
4. **Interpretability:** medium: additive trees, explained with permutation importance; per-prediction "why" bars in the app. Logistic Regression would be fully transparent but costs 0.10 PR-AUC.
5. **CV was honest:** test PR-AUC 0.786 vs CV 0.779.

Hyperparameters: `min_samples_leaf=80`, `max_leaf_nodes=31`, `max_iter=500`, `learning_rate=0.05`, `l2_regularization=1.0`

**Final test-set results** (single evaluation, threshold tuned on out-of-fold training predictions):

| Metric | Threshold 0.50 | Tuned threshold 0.59 |
|---|---:|---:|
| Accuracy | 0.8129 | 0.8325 |
| Precision | 0.6153 | 0.6688 |
| Recall | 0.8537 | 0.7752 |
| F1 | 0.7152 | 0.7181 |
| ROC-AUC | 0.9069 | 0.9069 |
| PR-AUC | 0.7856 | 0.7856 |

At 0.59 the model catches **3,720 of 4,799 cancellations** (77.5%), and **66.9%** of its alarms are right (1,842 false alarms among 12,643 kept bookings). The threshold maximises F1 on out-of-fold training predictions; a recall ≥ 80% policy would use 0.55.

| Confusion matrix (test) | Threshold tuning (out-of-fold) |
|---|---|
| ![Confusion matrix](reports/figures/05_confusion_matrix.png) | ![Threshold](reports/figures/09_threshold_tuning.png) |

## Explainability

Permutation importance of the final model on the test set (drop in PR-AUC when one input is shuffled):

1. **is_domestic**: Bookings with is_domestic = 1 cancel 36%, vs 24% when it is 0.
2. **lead_time**: Top quarter of lead_time (>= 125) cancels 38%, bottom quarter (<= 11) 10%.
3. **total_of_special_requests**: Bookings with 1+ total_of_special_requests cancel 22%, vs 33% with none.
4. **agent**: agent = 9 cancels 40% of the time, vs 6% for 28.
5. **market_segment**: market_segment = Online TA cancels 35% of the time, vs 12% for Corporate.
6. **required_car_parking_spaces**: Bookings with 1+ required_car_parking_spaces cancel 0%, vs 30% with none.

![Permutation importance](reports/figures/10_permutation_importance.png)

## Temporal validation (does it hold up next season?)

Train on arrivals **2015-07-01 → 2017-05-01**, test on **2017-05-02 → 2017-08-31** (17,544 bookings):

| Model | Random split PR-AUC | Temporal PR-AUC | Temporal ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.680 | 0.691 | 0.833 |
| Decision Tree | 0.477 | 0.474 | 0.662 |
| Random Forest | 0.776 | 0.683 | 0.830 |
| Random Forest (tuned) | 0.785 | 0.691 | 0.827 |
| HistGradientBoosting | 0.787 | 0.699 | 0.835 |
| HistGradientBoosting (tuned) | 0.786 | 0.703 | 0.837 |
| K-Nearest Neighbors | 0.675 | 0.611 | 0.778 |

The latest period cancels **34.7%** of bookings vs 25.7% before, and its months are under-represented in training. ROC-AUC drops for every model, and the tree ensembles lose the most: their lead over Logistic Regression shrinks from about 0.10 to about 0.01 PR-AUC, so part of what they learn is season-specific. That's real **data drift** that a random split hides. The selected model still ranks first. In production: retrain on a rolling window and re-check the threshold each season.

![Temporal vs random](reports/figures/13_temporal_vs_random.png)

## Web app

**CancelSense** (Flask + hand-written HTML/CSS/JS, Chart.js, D3, Plotly; light and dark mode;
responsive):

| Page | What it shows |
|---|---|
| **Overview** | Problem, dataset, pipeline, selected model, headline test metrics, top-3 leaderboard |
| **Model comparison** | Kaggle-style leaderboard with a "why it won" card, grouped bars (CV/test toggle), radar, sortable table, ROC + PR curves of all models, confusion matrix, head-to-head mode, fit/predict time and size, tuning scatter |
| **Overfitting** | Train vs CV vs test per model, learning curves, depth validation curve, gradient-boosting training replay (play/pause), decision-boundary playground (2 features x 5 models), temporal validation, leakage ablation |
| **Explainability** | Permutation importance + plain-English insights, tree importances, Logistic Regression weights, zoomable/collapsible D3 decision tree, "falling bookings" animation, rotatable 3-D PCA booking space |
| **Predict** | Full booking form with live what-if updates, animated risk gauge, "why this prediction" bars, what-if curves, random real test booking (with its true outcome), threshold slider, business-impact calculator |

A model switcher at the top of the analysis pages updates every chart for the chosen model.
Screenshots: [`screenshots/`](screenshots/).

## How to run

Requires Python 3.14 (same as Task 2). Windows PowerShell shown; on macOS/Linux use
`source .venv/bin/activate`.

```powershell
git clone https://github.com/Krishna141414/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt

python main.py               # full run, ~16 min: data -> compare -> tune -> select -> evaluate -> save
python main.py --quick       # ~7 min smoke test (tiny tuning budget)
python main.py --figures-only  # redraw reports/figures from the saved results

pytest                       # tests
ruff format --check . ; ruff check .

flask --app app.app run      # web app on http://127.0.0.1:5000
```

The dataset downloads automatically on the first run (verified to be 119,390 x 32) and is
also committed in `data/raw/`. The trained model and the web-app cache are committed too, so
the app runs straight after `pip install`, without training.

## Project structure

```
data/raw/hotel_bookings.csv   raw dataset (committed, read-only)
notebooks/modeling.ipynb      full walkthrough, executed with outputs
src/
  config.py                   every path, column list, hyperparameter, threshold and colour
  download_data.py            download + shape check (Task 2)
  data.py                     load + clean (Task 2's load_data.py + cleaning.py)
  features.py                 feature engineering + selection checks (Task 2) + preprocessors + splits
  models.py                   model zoo: every algorithm in a Pipeline with its preprocessing
  train.py                    CV, RandomizedSearchCV, selection, OOF threshold, curves, temporal, ablation
  evaluate.py                 metrics, curves, over/underfitting diagnosis, thresholds, importances
  insights.py                 tree JSON, falling-bookings paths, decision boundaries, PCA, boosting replay
  visualize.py                shared figure style + 16 figures
models/                       final_model.joblib + metrics.json
reports/                      model_comparison.csv + figures/
app/                          Flask app (app.py, services.py, build_cache.py, templates/, static/, cache/)
tests/                        pytest: data, features, preprocessing, models, train, evaluate, insights, app
docs/decisions.md             every decision: what, why, alternative considered
screenshots/                  (empty, for submission screenshots)
main.py                       one command for everything
render.yaml, gunicorn.conf.py deployment
```

## Tech stack

Python 3.14 · pandas 3.1 · NumPy 2.5 · scikit-learn 1.9 · matplotlib 3.11 · Flask 3.1 +
gunicorn · Chart.js 4.4, D3 7.9 and Plotly.js 2.35 (CDN, pinned) · pytest · ruff.
XGBoost/LightGBM are not used: HistGradientBoosting is scikit-learn's own LightGBM-style
booster, and skipping them keeps the deploy small (see decisions 5.5).

## Deploying on Render (free tier)

`render.yaml` is a Render Blueprint. Either use **New -> Blueprint** with this repo, or
create a **Web Service** by hand:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Branch | `main` |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app.app:app` |
| Health check path | `/healthz` |
| Instance type | Free |
| Environment variable | `PYTHON_VERSION` = `3.14.0` |

Nothing is trained on the server: it loads the committed 3 MB model and ~0.5 MB of JSON
(about 200 MB RAM in total, well under the 512 MB limit).

## Credits

- Dataset: Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.* Data in Brief, 22, 41-49.
- Data cleaning and feature engineering: my Task 2 project,
  [hotel-booking-preprocessing](https://github.com/Krishna141414/hotel-booking-preprocessing) (reused unchanged).
- Built by **Krishna** (B.Tech AI & ML) for the VoidMain Technologies AI Internship, Task 3.
