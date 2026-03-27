import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Example: "
        "postgresql+psycopg://user:password@localhost:5432/mushroom_feedback"
    )


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    input_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    prob_poisonous: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_borderline: Mapped[bool] = mapped_column(Boolean, nullable=False)

    feedback: Mapped["PredictionFeedback | None"] = relationship(
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )


class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"
    __table_args__ = (UniqueConstraint("prediction_id", name="uq_prediction_feedback_prediction_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    feedback_value: Mapped[str] = mapped_column(String(16), nullable=False)

    prediction: Mapped[Prediction] = relationship(back_populates="feedback")


def init_db():
    Base.metadata.create_all(bind=engine)
