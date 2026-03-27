from sqlalchemy import text

from db import SessionLocal


def clear_feedback_tables():
    with SessionLocal() as session:
        session.execute(
            text("TRUNCATE TABLE prediction_feedback, predictions RESTART IDENTITY CASCADE")
        )
        session.commit()

    print("Feedback-таблицы очищены.")


def main():
    clear_feedback_tables()


if __name__ == "__main__":
    main()
