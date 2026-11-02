# Agentic Loan Risk Triage System 

An end-to-end loan-risk assessment pipeline that turns a **free-text loan application / underwriting note** into a **model-backed risk explanation in plain English**.

Two LLM calls bookend a trained XGBoost classifier:

```
                 ┌─────────────────────┐
 terminal text → │ 1. LLM: extraction   │ → structured applicant record
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ 2. XGBoost + SHAP    │ → probability + top risk drivers
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
 terminal text ← │ 3. LLM: interpretation│ ← plain-language explanation
                 └─────────────────────┘
```

The client is a terminal CLI, the server is a FastAPI app, and the model is a locally trained XGBoost classifier — no data or model artifacts are sent to a third party; only the two LLM calls (via Hugging Face) leave the machine.

---

## How it works

1. **`client.py`** reads unstructured applicant text from the terminal (typed in, or from a file) and POSTs it to the server.
2. **`backend/server.py`** receives the text and runs it through four stages, calling functions defined in `src/`:
   1. `instruct_and_extract_schema` (`src/manual_llm_agent/huggingface_connect.py`) — an LLM call that extracts a structured, schema-validated applicant record (`CreditRiskApplicant`) from the free text.
   2. `extract_json` + `dict_to_rows` (`src/data_preprocess/process_text.py`) — turn that record into a single-row `DataFrame` with the exact column order and dtypes the model was trained on.
   3. `predict_with_shap` (`src/xgbmodel/models_predict.py`) — loads the trained XGBoost model, predicts a default probability, and uses SHAP to rank the features driving that prediction.
   4. `format_for_llm` + `instruct_and_interpret_report` (`src/manual_llm_agent/huggingface_connect.py`) — formats the prediction + SHAP drivers and sends them to a second LLM call that writes a short, borrower/loan-officer-facing explanation (no ML jargon, framed as a recommendation subject to human review).
3. The server returns `{ probability, explanation }`, and the client prints it back in the terminal.

---

## Project structure

```
loan_risk_project/
├── backend/
│   └── server.py                # FastAPI app: /health, /predict
├── client.py                    # Terminal CLI: health, assess
├── src/
│   ├── data_preprocess/
│   │   ├── process_text.py      # open_and_read_text, extract_json, dict_to_rows
│   │   └── feature_dtypes.json  # column order + dtypes the model expects
│   ├── xgbmodel/
│   │   └── models_predict.py    # load_model, predict_with_shap, report_tuple
│   └── manual_llm_agent/
│       ├── huggingface_connect.py  # CreditRiskApplicant schema + both LLM calls
│       └── HF_TOKEN.env            # HF_TOKEN=... (gitignored, create your own)
├── notebooks/
│   ├── data/loan_data.csv                # training data
│   ├── data_analysis/EDA.ipynb           # exploratory analysis
│   └── modelling/
│       ├── xgboost_train.ipynb           # training notebook (RandomizedSearchCV + 5-run check)
│       ├── xgboost_trained.json          # trained model artifact  <!-- NEW -->
│       ├── feature_dtypes.json           # dtypes written by the notebook  <!-- NEW -->
│       └── results/                      # saved metrics (see Model performance)  <!-- NEW -->
├── tests/
├── test_hosted_models/          # scratch scripts for testing HF endpoints directly
└── requirements.txt
```

---

## Setup

### 1. Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` is a full environment freeze; the packages this project actually relies on are:

| Purpose | Packages |
|---|---|
| API server | `fastapi`, `uvicorn`, `pydantic` |
| CLI client | `requests` |
| Modelling | `xgboost`, `scikit-learn`, `shap`, `pandas`, `numpy` |
| LLM agent | `langchain`, `langchain-core`, `langchain-huggingface`, `huggingface_hub`, `python-dotenv` |

### 2. Hugging Face token

The extraction and interpretation LLM calls run against a Hugging Face Inference Endpoint (`meta-llama/Llama-3.1-8B-Instruct`). Create `src/manual_llm_agent/HF_TOKEN.env` (this file is gitignored) with:

```
HF_TOKEN=hf_your_token_here
```

It's loaded automatically when `src/manual_llm_agent/huggingface_connect.py` is imported.

### 3. Trained model

<!-- NEW: the notebook now saves the model itself (it used `.save(...)`, which XGBClassifier doesn't have) -->
Run `notebooks/modelling/xgboost_train.ipynb` top to bottom. It saves the fitted `XGBClassifier` with:

```python
fit_model.save_model('xgboost_trained.json')
```

which writes `notebooks/modelling/xgboost_trained.json`. Point the server at it, either by editing `MODEL_PATH` in `backend/server.py` or with an environment variable:

```bash
# Windows (PowerShell)
$env:MODEL_PATH = "notebooks\modelling\xgboost_trained.json"
# macOS/Linux
export MODEL_PATH=notebooks/modelling/xgboost_trained.json
```

<!-- NEW: running the notebook headless -->
To run the notebook without opening Jupyter (needs `nbconvert` and a kernel registered for the venv):

```bash
python -m ipykernel install --user --name loanrisk-venv
cd notebooks/modelling
python -m nbconvert --to notebook --execute --inplace xgboost_train.ipynb --ExecutePreprocessor.kernel_name=loanrisk-venv
```

`feature_dtypes.json` (already checked in at `src/data_preprocess/feature_dtypes.json`) must match the column order/dtypes the model was trained on — the training notebook regenerates it automatically.

---

## Running it

Start the API server from the project root:

```bash
uvicorn backend.server:app --reload
```

Check it's up:

```bash
python client.py health
```

Run an assessment — type applicant details straight into the terminal:

```bash
python client.py assess
```

```
Enter the loan applicant details (finish with an empty line):
22-year-old female, Master's degree, renting, 0 years of employment experience,
$71,948 annual income, requesting a $35,000 personal loan at 16.02% APR,
credit score 561, 3-year credit history, no prior defaults.

Predicted probability of default/risk: 27.4%

This application is likely to require manual review before approval...
```

Or load applicant text from a file instead of typing it in:

```bash
python client.py assess --file applicant_note.txt
```

Point the client at a remote deployment:

```bash
python client.py --url http://your-host:8000 assess
```

---

## API reference

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

`model_loaded` is `false` if `MODEL_PATH` couldn't be loaded at startup — check the server logs.

### `POST /predict`

Request:

```json
{ "text": "free-text loan application or underwriting note" }
```

Response:

```json
{
  "probability": 0.274,
  "explanation": "This application is likely to require manual review... "
}
```

Errors (bad/unparseable text, model not loaded, etc.) come back as `HTTP 500` with a `detail` message.

---

## The applicant schema

`CreditRiskApplicant` (`src/manual_llm_agent/huggingface_connect.py`) is what the extraction LLM call is constrained to produce, and mirrors the training data's 13 input features:

| Field | Type | Notes |
|---|---|---|
| `person_age` | float | 18–100 |
| `person_gender` | enum | `male`, `female`, `other` |
| `person_education` | enum | `High School`, `Associate`, `Bachelor`, `Master`, `Doctorate` |
| `person_income` | float | annual, pre-tax |
| `person_emp_exp` | int | years |
| `person_home_ownership` | enum | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `loan_amnt` | float | requested amount |
| `loan_intent` | enum | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_int_rate` | float, optional | percent |
| `loan_percent_income` | float, optional | `loan_amnt / person_income` |
| `cb_person_cred_hist_length` | float | years |
| `credit_score` | int | 300–850 |
| `previous_loan_defaults_on_file` | enum | `Yes`, `No` |
| `loan_status` | int, optional | target only — always `null` at inference |

---

## Notebooks

- **`notebooks/data_analysis/EDA.ipynb`** — exploratory analysis of `loan_data.csv`.
- **`notebooks/modelling/xgboost_train.ipynb`**: splits the data 80/20 into train and test (stratified, `random_state=42`), then runs `RandomizedSearchCV` on the train part over an `XGBClassifier` (`enable_categorical=True`, `tree_method="hist"`, class-weighted for the ~3.5:1 negative:positive imbalance). It saves the best model to `xgboost_trained.json`, writes `feature_dtypes.json`, and saves all metrics to `results/`.  <!-- NEW: updated to match the notebook as it now runs -->

---

<!-- NEW SECTION: model performance -->
## Model performance

Data: `notebooks/data/loan_data.csv`, 45,000 rows, 22.2% class 1 (`loan_status = 1`).

### How the model is chosen

1. **Hold out a test set.** 20% of the data (9,000 rows) is set aside and not touched until the end.
2. **Search for hyperparameters.** `RandomizedSearchCV` tries 15 random settings. Each setting is scored with 5-fold stratified CV on the train part, using PR-AUC (`average_precision`). That's 75 fits.
3. **Refit the best setting** on the whole train part. This single model is `xgboost_trained.json`.
4. **Score it on train, val and test.**
5. **Check stability.** The best setting is retrained on 5 different random train/val/test splits (seeds 0–4) to see how much the scores move.

The search uses its 5 CV folds only to *choose* the hyperparameters. It returns one model. The 5-run check is a separate step afterwards.

### Best hyperparameters

| `n_estimators` | `max_depth` | `learning_rate` | `min_child_weight` | `subsample` | `scale_pos_weight` |
|---|---|---|---|---|---|
| 206 | 7 | 0.122 | 15 | 0.912 | 3.5 |

Best mean CV PR-AUC: **0.935 ± 0.003**. The top 5 settings all land between 0.933 and 0.935.

### Classification report: saved model (single split)

| split | rows | accuracy | precision (1) | recall (1) | f1 (1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| train | 30,600 | 0.957 | 0.848 | 0.983 | 0.911 | 0.995 | 0.983 |
| val | 5,400 | 0.914 | 0.766 | 0.885 | 0.821 | 0.976 | 0.933 |
| test | 9,000 | 0.921 | 0.777 | 0.902 | 0.835 | 0.978 | 0.940 |

On test, class 0 scores precision 0.97 and recall 0.93.

### Stability across 5 different splits (mean ± std)

| split | accuracy | precision (1) | recall (1) | f1 (1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| train | 0.957 ± 0.001 | 0.848 ± 0.003 | 0.983 ± 0.002 | 0.911 ± 0.002 | 0.995 ± 0.000 | 0.983 ± 0.000 |
| val | 0.916 ± 0.004 | 0.768 ± 0.012 | 0.892 ± 0.003 | 0.825 ± 0.007 | 0.977 ± 0.001 | 0.937 ± 0.004 |
| test | 0.918 ± 0.002 | 0.771 ± 0.004 | 0.898 ± 0.004 | 0.830 ± 0.003 | 0.978 ± 0.001 | 0.938 ± 0.002 |

What this shows:
- **The split hardly matters.** Every std is at most about 0.01. The largest is val precision (±0.012), which is expected because val is the smallest set.
- **There's mild overfitting.** Class-1 f1 is about 0.91 on train and about 0.83 on held-out data, and the gap is the same in every run.
- **The class weighting favours recall.** Class 1 has about 0.90 recall and about 0.77 precision, so the model catches most class-1 cases but raises some false alarms.

### Saved results (`notebooks/modelling/results/`)

| file | contents |
|---|---|
| `classification_reports.txt` | sklearn classification reports for train / val / test |
| `split_metrics.csv` | one row per split for the saved model |
| `cv_search_results.csv` | all 15 hyperparameter settings: mean/std CV PR-AUC and mean train score |
| `best_params.json` | full parameters of the saved model |
| `runs_metrics.csv` | 5 runs × 3 splits = 15 rows |
| `runs_summary_mean_std.csv` | mean ± std per split across the 5 runs |
| `runs_classification_reports.csv` | per-class precision/recall/f1/support for every run and split |
| `run1_classification_reports.txt` | reports from the first run, before the test split was seeded |

---

<!-- NEW SECTION: end-to-end agent test -->
## Agent test (`tests/extracted_text.txt`)

Input: the APEX National Bank underwriting note in `tests/extracted_text.txt`. The applicant is 22, female, has a Master's degree, rents, has 0 years of employment, earns $71,948, is requesting $35,000 for a personal loan at 16.02%, has a credit score of 561 and 3 years of credit history, and has no prior defaults.

| stage | status | time |
|---|---|---|
| 1. LLM extraction (`instruct_and_extract_schema`) | **failed**: Hugging Face `402 Payment Required` (no remaining Inference Provider credits) | — |
| 2. Preprocess (`extract_json` + `dict_to_rows`) | ok | ~10 ms |
| 3. XGBoost + SHAP (`predict_with_shap`) | ok | ~215 ms first call, ~235 ms afterwards (median of 5) |
| 4. LLM interpretation (`instruct_and_interpret_report`) | **not run**: same 402 error | — |

Stages 2–3 were run on a `CreditRiskApplicant` filled in by hand from the note, standing in for the extraction output. Model output (`xgboost_trained.json`):

```
Predicted probability (class 1): 99.9%
loan_percent_income=0.49           impact +3.281
previous_loan_defaults_on_file=No  impact +1.644
person_home_ownership=RENT         impact +1.291
loan_int_rate=16.02                impact +0.910
person_income=71948.0              impact +0.490
```

All the top drivers push towards class 1. A high loan-to-income ratio, renting and a high interest rate are classic risk signals, which fits the note's "requires manual underwriter review" recommendation. Credit score (561) isn't among the top drivers, which matches the training data: the class-1 rate is about 22% in every credit-score band.

Local model latency is about 0.25 s per request. End-to-end latency is set by the two LLM calls, which couldn't be measured until the HF account has credits again.

---

---

## Notes & caveats

- The `/predict` endpoint makes two live LLM calls per request (extraction + interpretation) in addition to the local model inference, so expect network latency and a valid `HF_TOKEN`.
- The Hugging Face Inference Endpoint used (`meta-llama/Llama-3.1-8B-Instruct`) must be reachable and warm on your Hugging Face account/token.
- **Label meaning is inconsistent.** `format_for_llm` describes the probability as "loan **approval**", but `client.py`, the examples above and the agent test read it as **default/risk**. In the data, `loan_percent_income` and renting raise the class-1 rate, which fits default risk. But every applicant with `previous_loan_defaults_on_file = Yes` has `loan_status = 0`, which fits approval. Check which meaning the dataset intends before relying on the wording in the LLM explanation.  <!-- NEW -->
- **Hugging Face credits:** the account in `HF_TOKEN.env` currently returns `402 Payment Required`, so `/predict` will fail with HTTP 500 until credits are added or the model is pointed at another provider.  <!-- NEW -->
- This tool produces a **model-based recommendation for human review**, not an automated approval/denial decision — the interpretation prompt explicitly frames it that way and flags any driver that could reflect a protected characteristic for fair-lending review.
