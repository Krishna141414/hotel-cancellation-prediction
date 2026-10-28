# Modelling Decisions Log

Every modelling choice in this project: **what** was decided, **why**, and the
**alternative** that was considered. Numbers come from `python main.py`
(`models/metrics.json`, `reports/model_comparison.csv`). Data cleaning and feature
engineering decisions were made in Task 2 and are summarised in section 2, with a link to the
full Task 2 log.

| Section | Topic |
|---|---|
| [1](#1-problem-and-metric) | Problem framing and primary metric |
| [2](#2-data-reused-from-task-2) | Data reused from Task 2 |
| [3](#3-leakage-review) | Leakage review (post-booking columns) |
| [4](#4-validation-setup) | Validation setup |
| [5](#5-models-and-preprocessing) | Models and preprocessing |
| [6](#6-hyperparameter-tuning) | Hyperparameter tuning |
| [7](#7-overfitting--underfitting-and-model-selection) | Over/underfitting and model selection |
| [8](#8-decision-threshold) | Decision threshold |
| [9](#9-explainability) | Explainability |
| [10](#10-final-evaluation-and-temporal-check) | Final evaluation and temporal check |
| [11](#11-web-app-and-deployment) | Web app and deployment |
| [12](#12-engineering) | Engineering |

---

## 1. Problem and metric

### 1.1 Binary classification of `is_canceled` at booking time
- **What:** predict whether a booking will be canceled, using only what is known when the
  booking is made.
- **Why:** this is the moment a revenue manager can act, by overbooking a likely-canceled
  room or asking for a deposit. Features recorded later (check-in, outcome date) are excluded
  (section 3).
- **Alternative:** predict cancellation timing (survival / regression on days before arrival).
  More useful but needs data the dataset does not have (the cancellation date leaks the outcome).

### 1.2 PR-AUC (average precision) is the primary metric
- **What:** models are ranked, tuned and selected on mean 5-fold CV PR-AUC. Accuracy,
  precision, recall, F1, ROC-AUC and the confusion matrix are reported alongside.
- **Why:** 27.5% of bookings are canceled. A model that always says "not canceled" is already
  72.5% accurate, so accuracy rewards ignoring the class we care about. PR-AUC summarises
  precision and recall of the *canceled* class over all thresholds, so it does not depend on
  the 0.5 cut-off (that is tuned separately, section 8). Its no-skill baseline is the
  positive rate (0.275), which makes "how much better than guessing" easy to read.
- **Alternative considered:** ROC-AUC. Also threshold-free, but it rewards ranking the large
  "not canceled" class well and looks optimistic under imbalance (the baseline is 0.5 and
  every real model is > 0.68). Both are shown everywhere. F1 was rejected as the primary
  metric because it depends on a threshold.

## 2. Data reused from Task 2

- **What:** `src/download_data.py`, `src/data.py` (= Task 2 `load_data.py` + `cleaning.py`) and
  the feature-engineering / selection part of `src/features.py` are copied from
  [hotel-booking-preprocessing](https://github.com/Krishna141414/hotel-booking-preprocessing)
  **with the logic unchanged**, together with their tests. The Task 2 repo is not imported at
  runtime: this repo runs on its own.
- **Result:** 119,390 raw rows -> 87,208 clean rows (exact duplicates, zero-guest rows,
  negative ADR, impossible counts removed; ADR capped at 335; structural missing values
  filled). Task 2 selected 33 inputs; this project drops `country` / `is_domestic` as a
  leakage precaution (3.5), leaving **31 model inputs** (18 numeric, 4 binary flags, 9 categorical).
- **Why reuse, not redo:** Task 2 already explored, cleaned and justified every step
  ([Task 2 decisions log](https://github.com/Krishna141414/hotel-booking-preprocessing/blob/main/docs/decisions.md)).
  Re-deriving it would only risk inconsistencies between the two submissions.
- **Small, documented adaptations** (none change which rows or values are produced):
  1. `reservation_status` / `reservation_status_date` are **dropped inside `clean_data`**
     (Task 2 kept them for EDA). Duplicates are still removed before they are dropped, so the
     row count is identical (87,208).
  2. The log1p column rule (skew > 1 and log halves |skew|) is now a transformer
     (`SkewLog1p`) that decides **inside `fit`**. Task 2 chose the columns once on the whole
     training set, which is fine for one fit but leaks a little across CV folds.
  3. A second, tree-friendly preprocessor (section 5.2) and a time-ordered split (4.4).
- **Known Task 2 caveat kept:** the ADR cap (99.9th percentile) is computed on the full
  cleaned data before the split. It moves 85 rows and barely depends on which rows are held
  out; this is noted rather than re-engineered.

## 3. Leakage review

Evidence comes from a **leakage ablation** (`src/train.py::leakage_ablation`): the final
model's 5-fold CV PR-AUC with suspected columns added or removed, on the training set only.

| Variant | Inputs | CV PR-AUC | vs final |
|---|---:|---:|---:|
| Selected features (final) | 31 | 0.7149 ± 0.0050 | +0.0000 |
| + booking_changes | 32 | 0.7219 ± 0.0049 | +0.0070 |
| + assigned_room_type | 32 | 0.7307 ± 0.0047 | +0.0158 |
| + both post-booking columns | 33 | 0.7374 ± 0.0052 | +0.0225 |
| - days_in_waiting_list | 30 | 0.7144 ± 0.0049 | -0.0005 |
| + country & is_domestic | 33 | 0.7785 ± 0.0059 | +0.0636 |

### 3.1 `reservation_status`, `reservation_status_date` -> dropped (hard leakage)
- They *are* the outcome (`Canceled` / `Check-Out` / `No-Show`) and its date. Task 2 measured a
  target purity of 1.000 for `reservation_status`.

### 3.2 `assigned_room_type` -> dropped
- **Why:** the room is (re)assigned at check-in, which canceled bookings never reach. A room
  that differs from the reserved one almost always means "the guest arrived" (Task 2: changed
  rooms cancel 4.7% vs 31.5%). It lifts CV PR-AUC by +0.0158, a gain that would vanish
  in production.
- **Alternative:** keep it for bookings already checked in. Not useful: those are not the
  ones we need to predict.

### 3.3 `booking_changes` -> dropped (the safer option)
- **Why:** it counts changes "from the moment the booking was entered until check-in or
  cancellation", a snapshot taken at the **outcome** date. A booking canceled after 3 days had
  3 days to collect changes, one kept for 200 days had 200. It adds +0.0070 CV PR-AUC,
  and together with `assigned_room_type` +0.0225.
- **Alternative considered:** keep it if predictions are refreshed daily with the change count
  *so far*. That needs a time-stamped change history the dataset does not have.

### 3.4 `days_in_waiting_list` -> kept
- **Why:** the days a booking waited *before it was confirmed* are known at confirmation, so it
  is a legitimate booking-time input. Removing it changes CV PR-AUC by only -0.0005, so
  it is neither leaky nor important.

### 3.5 `country` / `is_domestic` -> dropped (the safer option), kept as an ablation
- **What:** both columns are removed from the final model (`config.DROPPED_FEATURES`,
  `config.COUNTRY_COLUMNS`). The Predict page no longer asks for nationality.
- **Why:** canceled bookings never reach check-in. Some hotel systems only record or correct a
  guest's nationality *at check-in*, and the dataset paper does not say when `country` was
  filled in. If that's the case here, the column partly encodes "this guest showed up", which
  is the outcome. The data can't tell a real pattern from that artefact, and the pattern is
  suspiciously strong: Portuguese guests are 31% of bookings but 41% of
  cancellations (36% vs 24% cancel rate). Per the project rule ("if a decision
  could go either way, take the safer option"), the column is excluded so that every input
  is known when the booking is made.
- **What it costs:** the same final-model settings with and without the two columns
  (`main.py::country_comparison`, evaluation only; the drop was decided on the leakage
  argument above, not on these numbers):

  | HistGradientBoosting (tuned) | Without country (deployed, 31 inputs) | With country (33 inputs) | Difference |
  |---|---:|---:|---:|
  | CV PR-AUC | 0.7149 | 0.7785 | +0.0636 |
  | CV ROC-AUC | 0.8517 | 0.9038 | +0.0521 |
  | Test PR-AUC | 0.7210 | 0.7831 | +0.0621 |
  | Test ROC-AUC | 0.8553 | 0.9064 | +0.0511 |
  | Test F1 @ 0.54 | 0.6557 | 0.7187 | +0.0630 |
  | Test Recall @ 0.54 | 0.7062 | 0.8237 | +0.1175 |
  | Test Precision @ 0.54 | 0.6120 | 0.6375 | +0.0255 |

- **Alternative considered:** keep it (Task 2 did, and the first version of this project
  did). Worth revisiting if the hotel confirms nationality is entered at booking time: then
  the "with country" model is the better one.

## 4. Validation setup

### 4.1 Stratified 80/20 hold-out test set, used once
- **What:** `train_test_split(stratify=y, random_state=42)`, the same split as Task 2: 69,766
  train / 17,442 test rows, both 27.5% canceled. `main.py` touches the test set in a single
  step (8), **after** the model, its hyperparameters and the threshold are frozen.
- **Why:** an untouched test set is the only unbiased estimate once CV has been used for many
  decisions. Stratification keeps the class ratio identical in both sets.

### 4.2 Stratified 5-fold CV on the training set for every decision
- **What:** `StratifiedKFold(5, shuffle=True, random_state=42)`. The same folds are used for
  comparison, tuning, learning curves, the validation curve, threshold tuning (out-of-fold
  predictions) and the leakage ablation.
- **Why:** 5 folds give a mean and a spread (std) per model, so "better" can be separated from
  "lucky". Identical folds make models directly comparable.
- **Alternative:** 10 folds (smaller variance, twice the cost) or repeated CV. Fold std is
  already about 0.004 PR-AUC, much smaller than the gaps between models, so 5 is enough.

### 4.3 Train scores come from the CV training folds
- **Why:** "train vs CV" is then measured on exactly the same fits, which is the cleanest
  overfitting signal (`cross_validate(return_train_score=True)`).

### 4.4 Temporal validation (extra realism check)
- **What:** all clean bookings ordered by arrival date; train on everything before the
  80th-percentile arrival date (**2017-05-02**), test on later arrivals (17,544 rows).
  Whole days stay on one side.
- **Why:** a deployed model always predicts the *future*. The random split mixes seasons and
  years, so it cannot show drift.
- **Not a decision input:** the temporal experiment runs after selection and changes nothing.
  This keeps the model choice independent of it (see 10.2 for the results).

### 4.5 Class imbalance
- **What:** stratified splits and folds; `class_weight="balanced"` for Logistic Regression,
  Decision Tree, Random Forest and HistGradientBoosting; PR-AUC as the primary metric; a
  tuned threshold.
- **Why:** balanced weights stop models from settling for "never canceled" and make the 0.5
  default threshold meaningful for recall and F1 comparisons.
- **Alternative considered:** SMOTE / undersampling. Resampling must happen inside each fold
  (an extra dependency, imbalanced-learn) and creates synthetic bookings. Class weights give
  the same effect without inventing data.
- **Side effect:** class-weighted models output *scores* skewed towards "canceled" rather than
  calibrated probabilities. The threshold is tuned on those scores (section 8). Calibration
  (`CalibratedClassifierCV`) was considered and left out: the business decision only needs a
  well-chosen threshold, and calibration would add another fit to every model.

## 5. Models and preprocessing

### 5.1 The model zoo (`src/models.py`)
| Model | Why it is in the comparison | Starting settings |
|---|---|---|
| Dummy (most frequent) | Reference point: the score of "no model" | always predicts 0 |
| Logistic Regression | Simple, fast, interpretable linear baseline | C=1, balanced |
| Decision Tree | Readable rules; shows unconstrained overfitting | no depth limit, balanced |
| Random Forest | Strong bagged ensemble, robust default | 200 trees, sqrt features, balanced |
| HistGradientBoosting | State-of-the-art boosting in scikit-learn | 300 iters, lr 0.1, 31 leaves, balanced |
| K-Nearest Neighbors | Instance-based, very different inductive bias | k=25, distance weights |

### 5.2 Two preprocessing variants, both fitted inside the pipeline
- **"scaled"** (Logistic Regression, KNN): Task 2's transformer. Median impute -> log1p on
  skewed columns -> `StandardScaler` for numeric columns; flags as-is; categories one-hot with
  rare (< 1%) values grouped as "infrequent".
- **"tree"** (Decision Tree, Random Forest, HistGradientBoosting, Dummy): the same one-hot
  encoding but **no log or scaling**.
- **Why two:** distance- and gradient-based models need comparable scales; trees split on
  thresholds and do not care about monotonic transforms. Keeping raw units for trees makes
  every split readable ("lead_time <= 8 days"), which the interactive tree and the falling
  bookings animation rely on.
- **Alternative:** one shared scaled preprocessor for everything. It works, but the tree
  thresholds would be in standard-deviation units.

### 5.3 KNN trains on a stratified sample of 20,000 rows
- **Why:** KNN compares every prediction with every stored row, so 5-fold CV on 69,766 rows
  means about a billion distance computations per fold, and the stored model grows with the
  data. A 20,000-row stratified sample keeps it practical and is documented in the table
  (`training_rows`). Its test evaluation still uses the full test set.
- **Alternative:** the full training set with a ball tree. One-hot data has 86 dimensions,
  where trees bring little speed-up.

### 5.4 Unrestricted Decision Tree on purpose
- **Why:** an untuned tree shows clearly what overfitting looks like (train PR-AUC 1.0). The
  validation curve (7.4) then shows the depth that would fix it. Tuning it as well would only
  have mattered if it had been a top-2 model.

### 5.5 XGBoost / LightGBM skipped
- **Why:** the Windows wheel download for xgboost/lightgbm on Python 3.14 stalled for over 5
  minutes, and XGBoost's Linux wheel is large (with an NCCL dependency) for a 512 MB free
  instance. `HistGradientBoostingClassifier` is scikit-learn's histogram-based booster,
  inspired by LightGBM, so the algorithm family is covered with no extra dependency.

## 6. Hyperparameter tuning

### 6.1 RandomizedSearchCV on the top 2 models by CV PR-AUC
- **What:** the two best untuned models (**Random Forest and HistGradientBoosting**) are tuned with
  `RandomizedSearchCV` on the same 5 folds, recording every metric and train scores.
  Budget: Random Forest 12 settings x 5 folds = 60 fits (169 s); HistGradientBoosting 20 settings x 5 folds = 100 fits (193 s). Search spaces are in `config.PARAM_DISTRIBUTIONS`.
- **Why random search:** for the same budget it explores more values per important
  hyperparameter than a grid (Bergstra & Bengio, 2012); a full grid over 5 HGB parameters
  would be 1,280 settings x 5 folds.
- **Alternative:** Bayesian optimisation (Optuna). Better sample efficiency but another
  dependency; the gains here are small (6.3).

### 6.2 One-standard-error rule instead of "highest CV score"
- **What:** the refit setting is the one with the **smallest train-CV gap among settings
  whose CV score is within one fold-std of the best** (`one_standard_error_index`).
- **Why:** the first full run simply took the top-CV setting. For HistGradientBoosting it
  picked the most complex corner of the space (127 leaves, 500 iterations), raising the
  train-CV gap from 0.081 to 0.146 for +0.008 CV, about one fold-std (0.007). Differences
  inside one std are noise, so among them the less-memorising model is the better bet on new
  data (the same idea as the one-SE rule for regularisation paths).
- **Effect:** Random Forest: CV 0.7157 vs 0.7183 for the top setting (fold std 0.0045), train-CV gap 0.1299; HistGradientBoosting: CV 0.7149 vs 0.7203 for the top setting (fold std 0.0050), train-CV gap 0.0546.
- **Alternative:** plain argmax (more variance) or adding regularisation-only search spaces.

### 6.3 Tuning results
| Model | Untuned CV | Tuned CV | Chosen settings |
|---|---:|---:|---|
| Random Forest | 0.6956 | 0.7157 (top setting 0.7183) | `n_estimators=150`, `min_samples_leaf=8`, `max_features=0.2`, `max_depth=24` |
| HistGradientBoosting | 0.7147 | 0.7149 (top setting 0.7203) | `min_samples_leaf=40`, `max_leaf_nodes=63`, `max_iter=300`, `learning_rate=0.03`, `l2_regularization=0.1` |

- **Honest reading:** for HistGradientBoosting tuning adds almost no CV score (+0.0002); its
  value is the **smaller gap** (0.081 -> 0.055) from stronger regularisation (bigger leaves,
  lower learning rate). Random Forest gains more (+0.020) because its defaults grow trees to
  single bookings; bigger leaves and fewer features per split halve its gap (0.304 -> 0.130).
  The tuned models also hold up slightly better on the temporal split (10.2).

## 7. Overfitting / underfitting and model selection

### 7.1 Diagnosis rules (`src/evaluate.py::diagnose_fit`)
- **Overfitting:** train - CV gap above 0.05 PR-AUC ("mild"), above 0.10 ("strong").
- **Underfitting:** small gap, but CV more than 0.05 below the best model.
- **Why these numbers:** fold std is about 0.002-0.011, so 0.05 is far beyond noise. 0.05 below
  the best is roughly the difference between a linear and a non-linear model here.

### 7.2 Results
| Model | Train | CV (± std) | Test | Gap | Diagnosis |
|---|---:|---:|---:|---:|---|
| Dummy (most frequent) | 0.2751 | 0.2751 ± 0.0000 | 0.2751 | 0.0000 | Baseline |
| Logistic Regression | 0.6735 | 0.6728 ± 0.0077 | 0.6715 | 0.0007 | Good fit |
| Decision Tree | 0.9997 | 0.4214 ± 0.0016 | 0.4294 | 0.5783 | Strong overfitting |
| Random Forest | 0.9995 | 0.6956 ± 0.0042 | 0.7040 | 0.3039 | Strong overfitting |
| Random Forest (tuned) | 0.8456 | 0.7157 ± 0.0045 | 0.7232 | 0.1299 | Strong overfitting |
| HistGradientBoosting | 0.7957 | 0.7147 ± 0.0054 | 0.7220 | 0.0810 | Mild overfitting |
| HistGradientBoosting (tuned) | 0.7695 | 0.7149 ± 0.0050 | 0.7210 | 0.0546 | Mild overfitting |
| K-Nearest Neighbors | 1.0000 | 0.6132 ± 0.0113 | 0.6214 | 0.3868 | Strong overfitting |

- **Decision Tree (no depth limit):** memorises the training set (train 1.0) and scores far
  worse on CV. Each leaf holds one or a few bookings, so its probabilities are almost all 0 or
  1, which also hurts PR-AUC. **Strong overfitting.**
- **KNN:** distance weighting gives every training row weight 1 for itself (train 1.0). CV is
  well below the tree ensembles. **Strong overfitting** (and high variance).
- **Random Forest:** fully grown trees memorise too (train ~1.0), and averaging 200 of them
  still reaches a solid CV score (0.696). The gap is large but partly "benign" (it is how
  bagging works, and the test score matches CV). Tuning with bigger leaves lifts CV to 0.716
  and halves the gap.
- **Logistic Regression:** train = CV (gap ≈ 0), about 0.04 below the best model, just inside
  the 0.05 margin, so the rule labels it **good fit**. It sits on the underfitting border: a
  linear boundary cannot represent interactions such as "long lead time *and* no deposit
  *and* agent 9". Before country was dropped, the gap to the ensembles was 0.10 and it was
  labelled underfitting; nationality was one of the signals the trees combined best.
- **HistGradientBoosting:** the smallest gap among the strong models. Boosting adds trees
  only while they reduce the loss, and leaf-size / L2 limits keep each tree small.
- **Learning curves** (`07_learning_curves.png`): Logistic Regression is nearly flat
  (+0.012 CV from 10% to 100% of the data), so more data will not help a model this simple.
  The tree ensembles still gain with more rows (Random Forest +0.0055 on the last step), a
  variance-limited regime where more data or more regularisation helps.
- **Validation curve** (`08_validation_curve_depth.png`): Decision Tree CV PR-AUC peaks at
  depth 10; depth 1-3 underfits, beyond that the train score keeps rising while CV
  falls.

### 7.3 Model selection rule (`src/train.py::select_final_model`), CV evidence only
1. Candidates = models within **0.005 PR-AUC** of the best mean CV score (about one fold std,
   so not separable).
2. Remove candidates whose compressed file is above **50 MB** (512 MB free host).
3. Pick the smallest train-CV gap, then the smaller file, then the faster prediction.

### 7.4 Selected model: **HistGradientBoosting (tuned)**
- **Tied on CV** (within 0.005 of the best, 0.7157): Random Forest (tuned) (0.7157, gap 0.1299, 16.64 MB), HistGradientBoosting (tuned) (0.7149, gap 0.0546, 0.89 MB), HistGradientBoosting (0.7147, gap 0.0810, 0.44 MB).
- **Chosen: HistGradientBoosting (tuned)**: the smallest train-CV gap of the tied models (0.0546, mild overfitting), 0.89 MB on disk, 7 ms per 1,000 predictions, 5.7 s to train.
- **Confirmed once on the test set**: PR-AUC 0.7210 vs CV 0.7149, ROC-AUC 0.8553.

- **Alternative considered:** Random Forest (tuned). Statistically tied on CV (0.7157) and
  test, but its gap is larger, its file is 16.6 MB vs 0.89 MB, and it is slower
  to predict. On a 512 MB instance that is a real cost for no measurable gain.
- **Interpretability trade-off:** HistGradientBoosting is less transparent than a single tree
  or Logistic Regression. This is mitigated with permutation importance, per-prediction "why"
  bars, and the interpretable models shown side by side in the app. Logistic Regression would
  be the choice if every decision had to be explainable by a formula; it costs about 0.04
  PR-AUC on CV, and it even does slightly better on the temporal split (10.2), which makes it a
  credible fallback.

## 8. Decision threshold

- **What:** threshold **0.54**, chosen to **maximise F1 on out-of-fold training
  predictions** (`cross_val_predict` with the final pipeline).
- **Why out-of-fold:** each training row is predicted by a model that never saw it, so the
  threshold is tuned on honest predictions without touching the test set.
- **Why F1:** with no agreed cost figures, F1 balances "catch cancellations" (recall) against
  "don't raise false alarms" (precision). A **recall >= 80%** strategy is implemented too
  (`THRESHOLD_STRATEGY = "recall_target"`) and gives 0.43.
- **Effect (test):** at 0.50 precision 0.584 / recall 0.745 / F1 0.654; at 0.54 precision 0.612 / recall 0.706 / F1 0.656. A higher threshold means fewer false alarms at the price of a few more missed cancellations.
- **Business version:** the web app's calculator takes the hotel's own costs (empty room vs
  relocating an overbooked guest), picks the cost-minimising threshold on the out-of-fold
  predictions and reports the saving per 1,000 bookings on the test set. With the default
  €250 / €400 it suggests 0.78.

## 9. Explainability

### 9.1 Permutation importance on the test set, per raw input column
- **What:** drop in test PR-AUC when one input is shuffled (5 repeats), computed on the full
  pipeline so a categorical column (e.g. `agent`) is shuffled as one feature.
- **Why the test set:** importance should describe behaviour on unseen data; on training
  data an overfitting model also "relies" on noise. It is computed after selection and
  decides nothing.
- **Alternative:** SHAP. Richer per-row explanations, but a heavy extra dependency; the
  brief's permutation + tree importances cover global behaviour, and the app's per-prediction
  "why" bars cover local behaviour (9.3).

### 9.2 Main drivers (plain English)
1. **total_of_special_requests**: Bookings with 1+ total_of_special_requests cancel 22%, vs 33% with none.
2. **lead_time**: Top quarter of lead_time (>= 125) cancels 38%, bottom quarter (<= 11) 10%.
3. **previous_cancel_ratio**: Bookings with previous_cancel_ratio > 0 cancel 67%, vs 27% otherwise.
4. **market_segment**: market_segment = Online TA cancels 35% of the time, vs 12% for Corporate.
5. **agent**: agent = 9 cancels 40% of the time, vs 6% for 28.
6. **deposit_type**: deposit_type = Non Refund cancels 95% of the time, vs 27% for No Deposit.
7. **required_car_parking_spaces**: Bookings with 1+ required_car_parking_spaces cancel 0%, vs 30% with none.
8. **customer_type**: customer_type = Transient cancels 30% of the time, vs 15% for Transient-Party.

Impurity importances of the Decision Tree and Random Forest (`11_tree_feature_importance.png`)
agree with permutation importance on the main drivers: `lead_time`, special requests,
parking spaces, `agent` and `market_segment`. Impurity importance favours many-valued numeric
columns, which is why the single tree ranks `stay_value`, `arrival_date_day_of_month`,
`adr_per_person` and `adr` near the top; permutation importance shows those add little on
unseen data.

### 9.3 "Why this prediction" in the app
- **What:** for each input, the change in predicted probability when only that input is reset
  to the typical booking's value (median / most common).
- **Why:** it uses the model's own predictions (one batched call, about 25 rows), needs no
  extra library and is easy to explain. Inputs interact, so the bars are not additive; the
  app says so.

## 10. Final evaluation and temporal check

### 10.1 Test set (single evaluation)
| Metric | Threshold 0.50 | Tuned 0.54 |
|---|---:|---:|
| Accuracy | 0.7836 | 0.7960 |
| Precision | 0.5836 | 0.6120 |
| Recall | 0.7445 | 0.7062 |
| F1 | 0.6543 | 0.6557 |
| ROC-AUC | 0.8553 | 0.8553 |
| PR-AUC | 0.7210 | 0.7210 |

Confusion matrix at 0.54: 3,389 cancellations caught, 1,410 missed, 2,149 false alarms, 10,494 correct 'not canceled'.

CV and test agree within 0.0061 PR-AUC for the selected model, which confirms that the
validation setup gave an honest estimate.

### 10.2 Temporal validation
| Model | Random split PR-AUC | Temporal PR-AUC | Change | Temporal ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6715 | 0.6773 | +0.0058 | 0.7902 |
| Decision Tree | 0.4294 | 0.4286 | -0.0008 | 0.6165 |
| Random Forest | 0.7040 | 0.6399 | -0.0641 | 0.7783 |
| Random Forest (tuned) | 0.7232 | 0.6527 | -0.0705 | 0.7880 |
| HistGradientBoosting | 0.7220 | 0.6661 | -0.0559 | 0.7909 |
| HistGradientBoosting (tuned) | 0.7210 | 0.6704 | -0.0506 | 0.7925 |
| K-Nearest Neighbors | 0.6214 | 0.5718 | -0.0496 | 0.7348 |

Periods: train 2015-07-01 -> 2017-05-01 (69,664 rows), test 2017-05-02 -> 2017-08-31 (17,544 rows). The no-skill PR-AUC also rises from 0.257 to 0.347, so part of the change is the base rate itself; ROC-AUC (base-rate independent) tells the cleaner story.

- **Interpretation:** the latest period (May–Aug 2017) cancels 34.7% of bookings vs
  25.7% before, and its season is under-represented in training. The random split
  hides this drift. ROC-AUC (which ignores the base rate) drops for **every** model, and the
  tree ensembles drop the most. Their lead over Logistic Regression disappears: on the later
  period Logistic Regression has the best PR-AUC (0.677 vs 0.670 for the selected model), while
  the selected model keeps a marginally better ROC-AUC (0.793 vs 0.790). Part of the extra
  structure the ensembles learn is season-specific and does not carry over.
- **Why this does not change the selection:** the selection rule uses CV on the random split
  by design; the temporal check is a separate realism test (4.4). It does mean the expected
  real-world score is closer to 0.67 than 0.72, and that a simpler model is a serious
  candidate once drift is monitored. Worth re-testing with more seasons of data.
- **Recommendation:** retrain monthly or quarterly on a rolling window, re-check the
  threshold each season, and monitor the cancel rate of predictions.

## 11. Web app and deployment

### 11.1 Flask + hand-written HTML/CSS/JS
- **Why:** full control over layout and interactions; no Streamlit (ruled out). Charts use
  Chart.js 4.4.1, the tree and the falling-bookings animation D3 7.9.0, the 3-D view
  Plotly.js 2.35.2, all pinned on jsDelivr. Plotly loads lazily when the 3-D section scrolls
  into view.

### 11.2 Everything precomputed
- **What:** `main.py` writes `app/cache/dashboard.json`, `explore.json` and `predict.json`
  (about 0.5 MB together). The server loads them plus the < 1 MB final model; it never sees
  the dataset or trains anything.
- **Why:** Render's free instance has 512 MB RAM and sleeps when idle. The app uses about
  190 MB and starts in a few seconds.

### 11.3 Palette and accessibility
- **What:** midnight navy / off-white base, electric blue primary, pink = canceled / risk,
  mint teal = not canceled / good, marigold = selected model. The categorical palette was run
  through a colour-vision-deficiency validator: on white, the bright marigold `#FFB020` and
  mint `#22D3B6` are below 3:1 contrast, so **light mode and every PNG figure use deeper
  steps** (`#C98400`, `#0E9682`). Dark mode keeps the bright brand values (they pass contrast
  and CVD separation on `#141A33`). Model identity is never colour-only: legends, labels and
  tables always carry the name.

### 11.4 Features kept / skipped
All required pages and every "wow" feature were built: leaderboard, model switcher, threshold
slider, what-if simulator, falling bookings, decision-boundary playground, 3-D booking space,
training replay, head-to-head and business calculator. Nothing was skipped.

## 12. Engineering

- **No magic numbers:** every path, column list, hyperparameter, threshold, size limit and
  colour is in `src/config.py`.
- **Reproducible:** fixed `random_state = 42` everywhere, pinned versions, one command.
  `main.py` writes JSON with Unix newlines so outputs are identical across OSes.
- **Tests:** pytest covers data, features, preprocessing (fit-on-train-only), models (every
  pipeline fits; search spaces valid), training (selection and one-SE rules, OOF), evaluation
  (hand-checked metrics, diagnosis rules, thresholds), insights (tree export consistency) and
  the app (every page, every API section, validation errors).
- **Lint:** `ruff format` + `ruff check` (pycodestyle, pyflakes, isort, bugbear, pyupgrade,
  Google docstrings, type hints).
