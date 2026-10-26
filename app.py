from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from modules.data_processing import REQUIRED_COLUMNS, MODEL_FEATURES, load_uploaded_file, validate_dataframe, add_risk_label
from modules.exploratory_analysis import class_distribution, facility_counts, lead_time_chart, duration_participants_chart
from modules.prediction import load_pipeline, load_metadata, predict_one
from modules.dashboard import make_review_queue

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "facility_reservations_cleaned.csv"

st.set_page_config(page_title="Campus Facility Reservation Monitoring", page_icon="🏫", layout="wide")

@st.cache_data(show_spinner=False)
def read_default_data():
    return pd.read_csv(DEFAULT_CSV)

@st.cache_resource(show_spinner="Loading saved prediction pipeline...")
def get_model():
    return load_pipeline()

@st.cache_resource(show_spinner=False)
def get_model_metadata():
    return load_metadata()


def show_metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

st.sidebar.title("🏫 Facility Monitor")
st.sidebar.caption("Campus Facility Reservation Monitoring")
page = st.sidebar.selectbox("Navigate to", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.divider()
st.sidebar.caption("Model: DEMO-RF-0.1")
st.sidebar.warning("Demo baseline model. Original midterm model file was not supplied.")

# Keep the default dataset available and let users optionally replace it for this session.
if "facility_data" not in st.session_state:
    st.session_state.facility_data = read_default_data()

if page == "Home":
    st.title("Campus Facility Reservation Monitoring")
    st.subheader("Cancellation and No-Show Decision Support")
    st.write("This Streamlit application helps authorized facility administrators review reservation records, explore utilization patterns, and inspect model-generated risk flags before deciding on follow-up actions.")
    st.info("Important continuity note: the uploaded midterm notebook was for student academic risk, not facility reservations. The original facility midterm model was not provided. This package includes a clearly labelled provisional baseline model trained from the supplied facility dataset so the application can run and be tested.")
    df = st.session_state.facility_data
    status = df["Usage_Status"].astype(str).str.strip().str.lower() if "Usage_Status" in df.columns else pd.Series(dtype=str)
    at_risk_count = int(status.isin(["cancelled", "no-show", "no show"]).sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: show_metric("Reservations", f"{len(df):,}")
    with c2: show_metric("Used", f"{int((status == 'used').sum()):,}")
    with c3: show_metric("Cancelled", f"{int((status == 'cancelled').sum()):,}")
    with c4: show_metric("No-show", f"{int(status.isin(['no-show', 'no show']).sum()):,}")
    st.markdown("### Workflow")
    st.markdown("1. Upload or inspect reservation data.\n2. Validate required columns and review missing/duplicate values.\n3. Explore utilization and status patterns.\n4. Generate a risk prediction for a proposed reservation.\n5. Let an authorized administrator verify details and record the final decision outside this demo app.")
    st.caption("Predictions are advisory only. They must not automatically reject a reservation or restrict access to campus facilities.")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    st.write("Upload a CSV or Excel file using the same schema as the supplied facility reservation dataset. If no file is uploaded, the included cleaned dataset is used.")
    uploaded = st.file_uploader("Upload reservation data", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            incoming = load_uploaded_file(uploaded)
            errors, _ = validate_dataframe(incoming)
            if errors:
                for err in errors: st.error(err)
                st.warning("The uploaded file was not selected because it failed schema validation. The previous dataset remains active.")
            else:
                st.session_state.facility_data = incoming
                st.success(f"Loaded {len(incoming):,} rows from {uploaded.name} for this session.")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
    df = st.session_state.facility_data
    errors, missing = validate_dataframe(df)
    if errors:
        for err in errors: st.error(err)
    else:
        st.success("Schema check passed: all required columns are present.")
    st.markdown("### Data preview")
    st.dataframe(df.head(30), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1: show_metric("Rows", f"{len(df):,}")
    with c2: show_metric("Columns", f"{len(df.columns):,}")
    with c3: show_metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.markdown("### Missing values by column")
    missing_summary = df.isna().sum().rename("Missing Values").reset_index().rename(columns={"index":"Column"})
    st.dataframe(missing_summary, use_container_width=True, hide_index=True)
    st.download_button("Download current data as CSV", df.to_csv(index=False).encode("utf-8"), "facility_reservations_current.csv", "text/csv")
    with st.expander("Required schema"):
        st.code("\n".join(REQUIRED_COLUMNS))

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    df = st.session_state.facility_data
    st.markdown("### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    if "Usage_Status" in df.columns:
        st.plotly_chart(class_distribution(df), use_container_width=True)
    st.plotly_chart(facility_counts(df), use_container_width=True)
    if "Days_Before_Event" in df.columns and "Usage_Status" in df.columns:
        st.plotly_chart(lead_time_chart(df), use_container_width=True)
    if "Duration_Hours" in df.columns and "Participants" in df.columns and "Usage_Status" in df.columns:
        st.plotly_chart(duration_participants_chart(df), use_container_width=True)
    st.markdown("### Quick observations")
    if "Usage_Status" in df.columns:
        counts = df["Usage_Status"].value_counts(dropna=False)
        st.write(f"Most common recorded usage status: **{counts.index[0]}** ({int(counts.iloc[0])} records).")
    if "Facility_Type" in df.columns:
        counts2 = df["Facility_Type"].value_counts(dropna=False)
        st.write(f"Most frequently reserved facility type: **{counts2.index[0]}** ({int(counts2.iloc[0])} records).")
    st.caption("These are descriptive patterns in the supplied dataset, not proof that any field causes cancellations or no-shows.")

elif page == "Prediction":
    st.title("Reservation Risk Prediction")
    st.write("Enter information available before the reservation event. Post-event fields such as actual attendance, usage status, cancellation timing, and final outcome are intentionally excluded from prediction inputs.")
    df = st.session_state.facility_data
    pipeline = get_model()
    metadata = get_model_metadata()
    st.warning("This uses the provisional demo model, not the original midterm model. Replace the model artifact when the original facility model is available.")
    def choices(col, fallback):
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            return vals or fallback
        return fallback
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            user_type = st.selectbox("User type", choices("User_Type", ["Student", "Faculty", "Staff", "Student Organization"]))
            facility_type = st.selectbox("Facility type", choices("Facility_Type", ["Classroom", "Auditorium", "Laboratory", "Gymnasium", "Meeting Room"]))
            purpose = st.selectbox("Purpose", choices("Purpose", ["Class", "Meeting", "Seminar", "Training", "Other"]))
            after_hours = st.selectbox("After office hours?", choices("After_Office_Hours", ["No", "Yes"]))
        with c2:
            duration = st.number_input("Duration (hours)", min_value=1, max_value=24, value=2, step=1)
            participants = st.number_input("Expected participants", min_value=1, max_value=1000, value=30, step=1)
            days_before = st.number_input("Days before event when booked", min_value=0, max_value=365, value=7, step=1)
        submitted = st.form_submit_button("Predict reservation risk", type="primary")
    if submitted:
        row = {"User_Type": user_type, "Facility_Type": facility_type, "Duration_Hours": int(duration), "Participants": int(participants), "Days_Before_Event": int(days_before), "After_Office_Hours": after_hours, "Purpose": purpose}
        try:
            label, probability = predict_one(pipeline, row)
            st.markdown("### Prediction result")
            if label == "At Risk": st.error("Model result: AT RISK — possible cancellation or no-show")
            else: st.success("Model result: COMPLETED — lower risk according to this model")
            if probability is not None:
                st.metric("Estimated at-risk probability", f"{probability:.1%}")
            st.write("**Recommended human follow-up:** verify the request details and, if appropriate, confirm the reservation with the requester. Do not automatically deny the reservation based on this result.")
            st.caption(f"Model version: {metadata.get('model_version', 'unknown')} | Prediction timestamp: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
            with st.expander("Input fields used"):
                st.json(row)
        except Exception as exc:
            st.error(f"Prediction failed. Check that the saved pipeline matches the feature schema. Details: {exc}")
    with st.expander("Model evaluation metrics (provisional baseline)"):
        metric_path = ROOT / "models" / "evaluation_metrics.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            m2.metric("At-risk precision", f"{metrics['precision_at_risk']:.1%}")
            m3.metric("At-risk recall", f"{metrics['recall_at_risk']:.1%}")
            m4.metric("At-risk F1", f"{metrics['f1_at_risk']:.1%}")
            st.write("Confusion matrix labels:", metrics["confusion_matrix_labels"])
            st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=metrics["confusion_matrix_labels"], columns=metrics["confusion_matrix_labels"]), use_container_width=True)
            st.caption(metrics["model_source"])

elif page == "Dashboard":
    st.title("Facility Risk and Utilization Dashboard")
    df = st.session_state.facility_data
    pipeline = get_model()
    queue = make_review_queue(df, pipeline)
    if queue.empty:
        st.warning("No rows have all required pre-event prediction features.")
    else:
        risk_count = int((queue["Predicted_Risk"] == "At Risk").sum())
        total = len(queue)
        used = int((df.get("Usage_Status", pd.Series(dtype=str)) == "Used").sum()) if "Usage_Status" in df.columns else 0
        cancelled = int((df.get("Usage_Status", pd.Series(dtype=str)) == "Cancelled").sum()) if "Usage_Status" in df.columns else 0
        no_show = int((df.get("Usage_Status", pd.Series(dtype=str)).isin(["No-Show", "No Show"])).sum()) if "Usage_Status" in df.columns else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1: show_metric("Records scored", f"{total:,}")
        with c2: show_metric("Predicted at risk", f"{risk_count:,}")
        with c3: show_metric("Recorded cancellations", f"{cancelled:,}")
        with c4: show_metric("Recorded no-shows", f"{no_show:,}")
        st.plotly_chart(px.pie(queue, names="Predicted_Risk", title="Model-predicted risk split", hole=0.35), use_container_width=True)
        st.markdown("### Administrator review queue")
        risk_only = st.checkbox("Show only predicted at-risk reservations", value=True)
        view = queue[queue["Predicted_Risk"] == "At Risk"] if risk_only else queue
        cols = [c for c in ["Reservation_ID", "User_Type", "Facility_Type", "Purpose", "Days_Before_Event", "Predicted_Risk", "At_Risk_Probability", "Review_Status", "Recommended_Follow_Up"] if c in view.columns]
        st.dataframe(view[cols], use_container_width=True, hide_index=True)
        st.download_button("Download review queue CSV", view[cols].to_csv(index=False).encode("utf-8"), "facility_risk_review_queue.csv", "text/csv")
        st.markdown("### Human review record")
        st.write("This demo provides a review queue but does not persist official administrative decisions. Use your institution's authorized record system for final decisions and intervention logs.")
        with st.form("human_review_form"):
            reservation_ids = queue["Reservation_ID"].dropna().astype(str).tolist() if "Reservation_ID" in queue.columns else ["Unknown"]
            selected_id = st.selectbox("Reservation ID to review", reservation_ids)
            review_decision = st.selectbox("Administrator review", ["Pending", "Confirmed with requester", "Record verified", "Needs correction", "Follow-up completed", "Prediction disagreed with human review"])
            review_reason = st.text_area("Review notes / reason (do not enter sensitive personal data)")
            save_review = st.form_submit_button("Prepare review record")
        if save_review:
            record = {"Reservation_ID": selected_id, "Review_Status": review_decision, "Review_Notes": review_reason, "Review_Timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "Model_Version": get_model_metadata().get("model_version", "unknown")}
            st.session_state.last_review_record = record
            st.success("Review record prepared for download. This demo does not save it to a database.")
            st.json(record)
            st.download_button("Download review record JSON", json.dumps(record, indent=2).encode("utf-8"), f"review_{selected_id}.json", "application/json")

elif page == "System Design":
    st.title("Systems Analysis and Design")
    st.markdown("### Actors")
    st.markdown("- **Facility Administrator:** reviews predictions, checks reservation records, and records follow-up decisions.\n- **Reservation Requester:** provides reservation details and may confirm a booking when contacted.\n- **System Administrator:** maintains the application, model artifact, and authorized access.")
    st.markdown("### Functional requirements")
    fr = [
        ("FR-01", "The system shall accept CSV/XLSX reservation data and validate the required schema."),
        ("FR-02", "The system shall summarize missing values, duplicate rows, and dataset dimensions."),
        ("FR-03", "The system shall display descriptive statistics and at least three interactive EDA charts."),
        ("FR-04", "The system shall accept pre-event reservation details and show a model prediction with probability only when supported."),
        ("FR-05", "The system shall provide an at-risk review queue, model version/timestamp, and recommended human follow-up; final decisions remain human-controlled."),
    ]
    st.table(pd.DataFrame(fr, columns=["ID", "Requirement"]))
    st.markdown("### Non-functional requirements")
    nfr = [
        ("NFR-01 Privacy", "Only authorized personnel should access operational reservation data; do not publish identifiable or sensitive records in a public repository."),
        ("NFR-02 Performance", "For the provided 400-row dataset, the application should load the main dashboard and render charts within 5 seconds on a typical student laptop after model loading."),
        ("NFR-03 Reliability", "The application must validate uploads, show clear errors for invalid schemas, and record model version and prediction timestamp with prediction results."),
    ]
    st.table(pd.DataFrame(nfr, columns=["ID", "Requirement"]))
    st.markdown("### Analytical decision workflow")
    st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#496A81"]; A [label="Reservation request\n(pre-event fields)"]; B [label="Validate and preprocess"]; C [label="Saved model pipeline"]; D [label="Risk prediction\n+ timestamp/version"]; E [label="Administrator review queue"]; F [label="Verify details and\nrecord human decision"]; A -> B -> C -> D -> E -> F; }''')
    image_path = ROOT / "assets" / "use_case_diagram.png"
    if image_path.exists():
        st.markdown("### Use case diagram")
        st.image(str(image_path), use_container_width=True)
    st.markdown("### Prediction checkpoint and leakage control")
    st.write("Prediction is intended to happen before the event. Model inputs are User_Type, Facility_Type, Duration_Hours, Participants, Days_Before_Event, After_Office_Hours, and Purpose. Actual_Attendance, Cancellation_Days_Before, Usage_Status, and Reservation_Outcome are excluded from inference because they describe the event or its outcome and could leak future information.")
    st.markdown("### Continuity limitation")
    st.error("The uploaded student_performance.ipynb is for student academic risk, not facility reservations. No original facility midterm notebook or trained facility pipeline was provided. The included DEMO-RF-0.1 is a provisional baseline and must not be described as the original midterm model.")
    st.markdown("### Human accountability")
    st.write("A predicted risk is a prompt for review, not proof of a likely no-show and not automatic authority to reject a reservation. An authorized administrator should check source records, contact the requester when appropriate, and record the reason for any final decision.")

st.sidebar.divider()
st.sidebar.caption("Academic demo • Decisions must remain with authorized staff")
