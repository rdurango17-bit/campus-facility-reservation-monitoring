"""Model loading and inference helpers."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metadata.joblib"


def load_pipeline():
    return joblib.load(MODEL_PATH)


def load_metadata():
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {"model_version": "unknown", "model_source": "Unknown"}


def predict_one(pipeline, row):
    frame = pd.DataFrame([row])
    label = pipeline.predict(frame)[0]
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)[0]
        classes = list(pipeline.classes_)
        if "At Risk" in classes:
            probability = float(probabilities[classes.index("At Risk")])
    return str(label), probability


def predict_frame(pipeline, frame):
    result = frame.copy()
    result["Predicted_Risk"] = pipeline.predict(result).astype(str)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)
        classes = list(pipeline.classes_)
        if "At Risk" in classes:
            result["At_Risk_Probability"] = probabilities[:, classes.index("At Risk")]
    return result
