# Campus Facility Reservation Monitoring — Streamlit

A Streamlit decision-support demo for campus facility cancellation/no-show monitoring. It includes Home, Dataset, Data Analysis, Prediction, Dashboard, and System Design pages.

## Important model continuity note

The provided `student_performance.ipynb` notebook concerns **student academic risk**, not campus facility reservations. The original facility-reservation midterm notebook and saved model were not supplied. Therefore, this repository includes `models/trained_model.joblib` as a **provisional demo baseline** trained from the supplied cleaned facility dataset. It is not the original midterm model and must not be presented as such. Before final submission, replace the demo artifact and metrics with the instructor-approved original facility model and document any differences.

## Data and target definition

Included dataset: `data/facility_reservations_cleaned.csv` (converted from the supplied Excel file).

For the provisional baseline only, the label is derived from `Usage_Status`:
- `At Risk`: `Cancelled` or `No-Show`
- `Completed`: `Used`

The model uses only fields intended to be known before the event: `User_Type`, `Facility_Type`, `Duration_Hours`, `Participants`, `Days_Before_Event`, `After_Office_Hours`, and `Purpose`.

The model intentionally excludes `Actual_Attendance`, `Cancellation_Days_Before`, `Usage_Status`, and `Reservation_Outcome` from inference because these fields describe event execution or the final outcome and can cause target leakage. Check this target and feature definition against the original midterm submission before finalizing.

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The model artifact and evaluation metrics are included so the app can start without training at runtime. To regenerate the **demo baseline** from the included CSV:

```bash
python train_baseline_model.py
```

This script trains once and saves the preprocessing + model pipeline with joblib. Do not describe these results as the original midterm results.

## Pages

1. **Home** — purpose, summary KPIs, decision workflow.
2. **Dataset** — CSV/XLSX upload, schema validation, preview, missing-value and duplicate summaries, CSV download.
3. **Data Analysis** — descriptive statistics and interactive charts.
4. **Prediction** — pre-event input form, risk class, supported probability, timestamp/version, provisional model metrics.
5. **Dashboard** — scored review queue, filter, CSV export, human-review record export.
6. **System Design** — actors, five functional requirements, three measurable non-functional requirements, workflow, use-case diagram, leakage note, and human accountability.

## Deployment on Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. Confirm `app.py`, `requirements.txt`, `modules/`, `models/`, `assets/`, and `data/` are at the repository root.
3. Open Streamlit Community Cloud, sign in, choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy and inspect the app URL and logs.
5. Test all six pages, CSV/XLSX upload, invalid schema handling, predictions, and review-record download.

Do not publish real sensitive records in a public repository. Use synthetic or properly authorized data. Deployment must be completed from your own GitHub/Streamlit account; this package does not contain a deployed URL.

## Test checklist

- [ ] App opens without an error.
- [ ] All six pages work.
- [ ] Valid CSV/XLSX uploads pass validation.
- [ ] Missing columns and empty data produce useful errors.
- [ ] At least three interactive charts render.
- [ ] Prediction uses only the pre-event feature columns.
- [ ] Model version and prediction timestamp are visible.
- [ ] Review queue and review record export work.
- [ ] Original midterm model is substituted and same-case outputs are checked before claiming model continuity.

## Limitations

This is a classroom decision-support prototype, not a production reservation system. It does not implement authentication, a database-backed audit log, or persistent review decisions. Model predictions are not proof and must not automatically reject reservations. Model metrics are based on a held-out split of the supplied dataset and should not be treated as evidence of real-world performance.


## Additional submission materials
- `revised_facility_reservation_demo.ipynb` — revised reproducible baseline notebook (not the original midterm notebook).
- `sample_predictions.csv` — five sample predictions from the provisional model.
- `DECISION_SUPPORT_WORKFLOW.md` — workflow documentation.
- `SUBMISSION_CHECKLIST.md` — final submission status and URL placeholders.
- `screenshots/README_CAPTURE_SCREENSHOTS.txt` — required real screenshot checklist.

**Still required from the student:** the original facility-reservation midterm notebook/pipeline, genuine screenshots captured from the running app, and GitHub/Streamlit deployment URLs.
