import pandas as pd

from db import Prediction, PredictionFeedback, SessionLocal


NUM_FEATURES = ["cap-diameter", "stem-height", "stem-width"]
CAT_FEATURES = [
    "cap-shape",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-color",
    "stem-color",
    "ring-type",
    "habitat",
    "season",
    "has-ring",
]
FEATURE_COLS = CAT_FEATURES + NUM_FEATURES


def load_feedback_dataframe():
    with SessionLocal() as session:
        rows = (
            session.query(
                Prediction.input_payload,
                Prediction.label,
                PredictionFeedback.feedback_value,
            )
            .join(
                PredictionFeedback,
                Prediction.id == PredictionFeedback.prediction_id,
            )
            .all()
        )

    feedback_rows = []
    for input_payload, predicted_label, feedback_value in rows:
        true_label = predicted_label
        if feedback_value == "incorrect":
            true_label = "e" if predicted_label == "p" else "p"

        row = dict(input_payload)
        row["class"] = true_label
        feedback_rows.append(row)

    if not feedback_rows:
        return pd.DataFrame(columns=FEATURE_COLS + ["class"])

    feedback_df = pd.DataFrame(feedback_rows)
    for col in FEATURE_COLS:
        if col not in feedback_df.columns:
            feedback_df[col] = pd.NA

    return feedback_df[FEATURE_COLS + ["class"]]


def load_feedback_train_parts():
    feedback_df = load_feedback_dataframe()
    X_feedback = feedback_df[FEATURE_COLS].copy()
    y_feedback = feedback_df["class"].copy()
    return X_feedback, y_feedback
