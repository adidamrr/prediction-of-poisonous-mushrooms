from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from db import Prediction, PredictionFeedback, SessionLocal, init_db
from predict import load_artifacts, model_predict


REVIEW_MARGIN = 0.07


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Mushroom Safety Screening API", lifespan=lifespan)
model, meta = load_artifacts()


class MushroomRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cap_diameter: float = Field(alias="cap-diameter")
    stem_height: float = Field(alias="stem-height")
    stem_width: float = Field(alias="stem-width")
    cap_color: str = Field(alias="cap-color")
    stem_color: str = Field(alias="stem-color")
    cap_shape: str | None = Field(default=None, alias="cap-shape")
    does_bruise_or_bleed: str | None = Field(default=None, alias="does-bruise-or-bleed")
    gill_color: str | None = Field(default=None, alias="gill-color")
    ring_type: str | None = Field(default=None, alias="ring-type")
    habitat: str | None = None
    season: str | None = None
    has_ring: str | None = Field(default=None, alias="has-ring")


class FeedbackRequest(BaseModel):
    prediction_id: int
    feedback_value: Literal["correct", "incorrect"]


def get_decision(probability):
    threshold = meta["threshold"]
    if probability >= threshold:
        return "unsafe", "Есть риск токсичности. Не употребляйте гриб без экспертной проверки."
    if probability >= max(0.0, threshold - REVIEW_MARGIN):
        return "review", "Случай пограничный. Модель не уверена, нужна дополнительная проверка."
    return "likely_edible", "Вероятность токсичности ниже порога, но результат не гарантирует безопасность."


def is_borderline(probability, threshold):
    return abs(probability - threshold) <= threshold * 0.1


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict") 
def predict(payload: MushroomRequest):
    input_payload = payload.model_dump(by_alias=True, exclude_none=True)
    X = pd.DataFrame([input_payload])
    labels, probabilities = model_predict(X, model=model, meta=meta)

    probability = float(probabilities[0])
    label = labels[0]
    decision, message = get_decision(probability)
    threshold = meta["threshold"]
    borderline = is_borderline(probability, threshold)

    with SessionLocal() as session:
        prediction = Prediction(
            input_payload=input_payload,
            prob_poisonous=probability,
            label=label,
            decision=decision,
            threshold=threshold,
            is_borderline=borderline,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)

    return {
        "prediction_id": prediction.id,
        "prob_poisonous": probability,
        "label": label,
        "decision": decision,
        "message": message,
        "threshold": threshold,
        "is_borderline": borderline,
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    with SessionLocal() as session:
        prediction = session.get(Prediction, payload.prediction_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction not found")

        existing_feedback = (
            session.query(PredictionFeedback)
            .filter(PredictionFeedback.prediction_id == payload.prediction_id)
            .first()
        )
        if existing_feedback is not None:
            raise HTTPException(status_code=409, detail="Feedback already exists for this prediction")

        feedback_item = PredictionFeedback(
            prediction_id=payload.prediction_id,
            feedback_value=payload.feedback_value,
        )
        session.add(feedback_item)
        session.commit()
        session.refresh(feedback_item)

    return {
        "status": "ok",
        "feedback_id": feedback_item.id,
    }
