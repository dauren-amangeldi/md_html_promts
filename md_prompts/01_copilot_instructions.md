

## 📄 01_copilot_instructions.md

````markdown
# Copilot Instructions: Build PD Model Report from Notebooks

You are an assistant that generates a **comprehensive, well-structured HTML/Markdown report** for a credit risk PD model project, based on the following Jupyter notebooks:

- `data_preprocess_v4.ipynb` — data collection and feature engineering.
- `model_v4_approval_classification.ipynb` — first behavioral approach with RMSE(PD_approval, predict_proba) minimization at LOAN_AGE_MONTHS = 0.
- `model_v3.1_classification.ipynb` — main behavioral PD model.
- `movel_v3.1_regression.ipynb` — regression model approximating PD_approval.
- `movel_v3.1_hybrid.ipynb` — hybrid PD model (initial test parameters k and x0).
- `optimize_hybrid_params.ipynb` — Optuna-based optimization of k and x0.

Your job is to:

1. **Read and analyze** these notebooks.
2. **Extract key logic, metrics, and plots.**
3. **Fill in the report template** in `02_report_template.md`.
4. Produce:
   - `report.md` — final report in Markdown (RU),
   - `report.html` — HTML version with **Plotly interactive charts** embedded.

---

## 1. Project context (PD modelling)

The project is about modelling **Probability of Default (PD)** for a specific loan product with:

- **Behavioral PD model** (main):
  - Predicts `default_90plus_next_month` (90+ DPD) one month ahead.
  - Uses monthly behavioral features and loan age (`loan_age_months`).
  - Implemented in `model_v3.1_classification.ipynb`.

- **Application / approval PD (PD_approval)**:
  - Existing scoring PD at origination.
  - Used as a reference on `loan_age_months = 0`.
  - Approximated by `movel_v3.1_regression.ipynb` for historical periods without PD_approval.

- **First attempt** (in `model_v4_approval_classification.ipynb`):
  - Behavioral model trained with additional constraint:
    - try to minimize `RMSE(PD_approval, predict_proba)` for `LOAN_AGE_MONTHS = 0`.
  - This attempt is considered **methodologically problematic** and rejected, but must be described in the report.

- **Hybrid model**:
  - Implemented in `movel_v3.1_hybrid.ipynb`.
  - Combines `PD_app` and `PD_beh` using a logistic weight depending on `loan_age_months`.
  - Parameters `k` and `x0` are optimized in `optimize_hybrid_params.ipynb` via Optuna.

---

## 2. Hybrid PD formula (FIXED, do not alter)

Weight function:

```python
def weight(loan_age_months: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + math.exp(k * (loan_age_months - x0)))
````

Final PD:

```python
PD_final = w * PD_app + (1 - w) * PD_beh
```

Where:

* `PD_app`:

  * PD at origination,
  * from existing scoring system or approximated via regression in `movel_v3.1_regression.ipynb`.
* `PD_beh`:

  * Behavioral PD from `model_v3.1_classification.ipynb`.
* `w = weight(loan_age_months, k, x0)`.

In the report, you must:

* Explicitly show this formula.
* Explain the economic intuition:

  * at age ≈ 0 → mostly Application PD,
  * with increasing `loan_age_months` → more weight on Behavioral PD.

---

## 3. What you must do step-by-step

### 3.1. Analyze notebooks

For each notebook:

1. `data_preprocess_v4.ipynb`

   * Extract:

     * data sources,
     * time window,
     * filters applied,
     * final dataset size,
     * key feature engineering steps.
   * Identify:

     * how `loan_age_months` is computed,
     * how the target (90+ DPD next month) is constructed.

2. `model_v4_approval_classification.ipynb`

   * Summarize:

     * that this notebook represents the **first approach**:

       * behavioral classification model,
       * with additional minimization of `RMSE(PD_approval, predict_proba)` for `LOAN_AGE_MONTHS = 0`.
   * Extract:

     * model type (algo),
     * features,
     * main metrics,
     * key conclusion why this approach is **rejected**:

       * conflict between predicting TARGET and forcing agreement with PD_approval.

3. `model_v3.1_classification.ipynb`

   * This is the main **Behavioral PD model**.
   * Extract:

     * target definition and filtering,
     * model type / hyperparameters,
     * handling class imbalance,
     * metrics:

       * ROC-AUC (expect ~0.92),
       * Recall (~0.67),
       * KS, Gini, etc., if available.
   * Generate or reuse:

     * ROC curve (Plotly),
     * distribution of PD_beh,
     * possibly calibration plot.

4. `movel_v3.1_regression.ipynb`

   * This is a **regression model approximating PD_approval**:

     * input: scoring data (`score_bal`, `xgboost_used_flag`, `score_category`, etc.),
     * output: `pd_approval`.
   * Extract:

     * model type,
     * metrics (R², RMSE/MAE),
     * how well regression approximates PD_approval.
   * This must be described as a way to reconstruct / backfill PD_approval for older periods.

5. `movel_v3.1_hybrid.ipynb`

   * Implementation of the hybrid PD logic.
   * Extract:

     * how `PD_app` is sourced (real vs regressed),
     * how `PD_final` is computed,
     * how test values of `k` and `x0` affect:

       * PD distributions,
       * PD by `loan_age_months`.

6. `optimize_hybrid_params.ipynb`

   * Extract:

     * Optuna objective function,
     * criteria for optimizing `k` and `x0`:

       * proximity of `PD_final(age=0)` to `PD_approval`,
       * smooth or monotone average PD_final by `loan_age_months`,
       * preservation of ROC-AUC / Gini.
   * Extract final best values for `k` and `x0`.
   * Include them in the report.

---

### 3.2. Extract key metrics and build tables

From the notebooks, gather metrics for:

* Behavioral model (`model_v3.1_classification.ipynb`).
* Hybrid model (`movel_v3.1_hybrid.ipynb` with optimized `k`, `x0`).
* Application PD model (if metrics exist) or at least summary of its quality.

Prepare tables for section **6.1** in `02_report_template.md`:

* ROC-AUC, Gini, KS, Recall/Precision for:

  * Behavioral PD,
  * Hybrid PD,
  * Application PD (if applicable).

---

### 3.3. Generate Plotly charts

For the HTML report:

* Build Plotly figures (inside notebooks or separate scripts) and export HTML snippets via:

  ```python
  fig.to_html(full_html=False, include_plotlyjs=False)
  ```

* Relevant charts (to be embedded in `report.html`):

  1. ROC curve of Behavioral PD.
  2. ROC curve comparison (Application vs Behavioral vs Hybrid), if possible.
  3. Distribution of `PD_beh`, `PD_app`, `PD_final`.
  4. Average PD_by_age:

     * For each model:

       * `mean(PD_app | loan_age_months = 0)` (if applicable),
       * `mean(PD_beh | loan_age_months)`,
       * `mean(PD_final | loan_age_months)`.

* Embed resulting `<div>...</div>` blocks into placeholders in `02_report_template.md` → `report.html`.

---

## 4. Fill the template `02_report_template.md`

* Open `02_report_template.md`.
* For each `<!-- COPILOT: ... -->` block:

  * Insert synthesized text (in Russian) using real info from notebooks.
* Replace dummy placeholders (e.g. product name, dates, metrics) by concrete values.

Style:

* Russian, formal-business.
* Audience:

  * Deputy Chairman,
  * risk management,
  * validation / internal audit.
* Avoid "chatty" tone in the report; write as an internal model documentation / validation report.

---

## 5. Output files

At the end, you must produce:

1. `report.md`

   * Markdown version of the filled PD model report.

2. `report.html`

   * HTML version with:

     * embedded Plotly JS (once, via CDN),
     * embedded Plotly figure HTML snippets in the corresponding sections.

Make sure `report.html` is self-contained (except for Plotly CDN).

````
