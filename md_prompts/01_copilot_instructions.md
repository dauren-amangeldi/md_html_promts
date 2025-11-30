
## 📝 Файл 1: `01_copilot_instructions.md`

````markdown
# Copilot Instructions: Build PD Model Report

You are an assistant that generates a **comprehensive, well-structured HTML/Markdown report** for a credit risk PD model project.

## Goals

- Read and analyze **all relevant files** in this repository:
  - Python modules (src, models, utils, etc.)
  - Jupyter notebooks (EDA, model training, validation)
  - Config files with hyperparameters and model paths
  - Metrics / logs / CSVs with results
- Use this information to **fill in the report template** in `02_report_template.md`.
- Produce:
  1. `report.md` — human-readable report in Markdown.
  2. `report.html` — HTML version of the report with **Plotly interactive charts** embedded.

## High-level rules

- The project is about **Probability of Default (PD) modelling** for a specific loan product.
- There are at least two key models:
  - **Application (score) PD model**: PD at origination, based on application data.
  - **Behavioral PD model**: PD (90+ DPD) one month ahead, based on loan behavior.
- There is also a **hybrid model**:
  - Combines `PD_app` and `PD_beh` via a logistic weight function depending on `loan_age_months`.
  - Weight formula (IMPORTANT):

    ```python
    def weight(loan_age_months: float, k: float, x0: float) -> float:
        return 1.0 / (1.0 + math.exp(k * (loan_age_months - x0)))
    ```

    where:
    - `k` — steepness parameter (controls transition speed),
    - `x0` — midpoint where contribution of application vs behavioral is 50/50.

  - Final PD is:

    ```python
    PD_final = w * PD_app + (1 - w) * PD_beh
    ```

## What you must do step-by-step

1. **Locate data and models**
   - Find where the following live:
     - Application PD / score model training code and metrics.
     - Behavioral PD model training code and metrics.
     - Hybrid combination logic (look for `weight`, `loan_age_months`, `PD_app`, `PD_beh`).
   - Note the main file paths to reference them later in the report.

2. **Extract key metrics**
   For each model (Application, Behavioral, Hybrid):
   - ROC-AUC / Gini, KS, Recall / Precision, etc.
   - OOT test metrics, if available.
   - Calibration plots or Brier score, if available.
   - Stability metrics, if present.

   Store these numbers as tables to insert into the report.

3. **Generate Plotly charts**
   Prepare code snippets or directly generate (if running within a notebook) the following charts:

   - Distribution of PDs by model (Application vs Behavioral vs Hybrid).
   - PD vs `loan_age_months`:
     - plot **average PD per age bucket** for each model;
     - especially highlight Hybrid PD:
       - show that average `PD_final` by `loan_age_months` behaves logically (non-decreasing or business-consistent).
   - ROC curves for the models.
   - Optional: calibration curves (predicted PD vs observed default rate).

   When embedding charts in HTML, use:

   ```python
   fig.to_html(full_html=False, include_plotlyjs='cdn')
````

or similar Plotly methods to get `<div>` blocks to include into `report.html`.

4. **Fill the template `02_report_template.md`**

   * Open `02_report_template.md`.
   * For each section marked with `<!-- COPILOT: ... -->`, fill in content based on:

     * actual code,
     * actual metrics,
     * real plots (referenced as embedded Plotly graphs or images).
   * Be concise, but professional. The audience includes:

     * Deputy Chairman,
     * risk managers familiar with machine learning,
     * model validation / audit teams.

5. **Export Markdown and HTML**

   * Save the filled report as `report.md`.
   * Convert it to HTML (either via:

     * direct Markdown to HTML conversion, or
     * building an HTML skeleton and embedding Markdown content).
   * Ensure that Plotly charts are embedded as interactive elements:

     * include Plotly JS via CDN once at the top of the HTML,
     * then embed all figures’ HTML blocks in appropriate sections.

## Writing style for the report

* Write in **Russian**.
* Style: формально-деловой, понятный для:

  * риск-менеджеров,
  * руководства,
  * внутреннего аудита.
* Избегать жаргона кода в основном тексте отчёта, но давать ссылки на файлы:

  * например: `см. src/models/behavioral_model.py`.

## Important constraints

* Do **not invent** metrics or graphs: use only what you find in the repo.
* If чего-то нет (например, нет calibration plot):

  * честно указать это в отчёте,
  * предложить как «рекомендацию» на будущее.
* Строго сохранять формулу гибридной модели и её интерпретацию.

## Output summary

At the end you must produce:

* `report.md` — main report in Markdown.
* `report.html` — main report in HTML with Plotly charts embedded.

Use `02_report_template.md` as the backbone structure.

````
