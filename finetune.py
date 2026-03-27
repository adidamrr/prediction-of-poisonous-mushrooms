import json
import os

import numpy as np
from catboost import CatBoostClassifier, Pool

from clear_feedback import clear_feedback_tables
from feedback_dataset import load_feedback_dataframe


def main():
    os.makedirs("model", exist_ok=True)

    feedback_df = load_feedback_dataframe()
    if feedback_df.empty:
        print("В базе нет feedback-данных для дообучения.")
        return

    base_model = CatBoostClassifier()
    base_model.load_model("model/model.cbm")

    with open("model/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    nan_token = meta["nan_token"]

    feature_cols = list(base_model.feature_names_)
    cat_feature_names = [col for col in feature_cols if col in cat_cols]
    X_feedback = feedback_df[feature_cols].copy()
    y_feedback = (feedback_df["class"].astype(str) == "p").astype(int)

    X_feedback[num_cols] = X_feedback[num_cols].astype(float)
    X_feedback[num_cols] = np.log1p(X_feedback[num_cols])
    X_feedback[cat_cols] = X_feedback[cat_cols].fillna(nan_token).astype(str)

    feedback_pool = Pool(X_feedback, y_feedback, cat_features=cat_feature_names)

    finetuned_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=10,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=8,
        random_strength=1.5,
        od_type="Iter",
        od_wait=200,
        verbose=20,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    finetuned_model.fit(feedback_pool, init_model=base_model)

    finetuned_model.save_model("model/finetuned_model.cbm")
    clear_feedback_tables()


if __name__ == "__main__":
    main()
