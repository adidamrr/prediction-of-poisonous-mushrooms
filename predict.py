import json
import numpy as np
from catboost import CatBoostClassifier, Pool


def load_artifacts():
    model = CatBoostClassifier()
    model.load_model("model/model.cbm")

    with open("model/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def prepare_data(X, meta):
    X_copy = X.copy()
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    nan_token = meta["nan_token"]

    for col in num_cols + cat_cols:
        if col not in X_copy.columns:
            X_copy[col] = np.nan

    X_copy = X_copy[num_cols + cat_cols]
    X_copy[num_cols] = np.log1p(X_copy[num_cols])
    X_copy[cat_cols] = X_copy[cat_cols].fillna(nan_token).astype(str)

    return X_copy


def model_predict(X_input, model=None, meta=None):
    if model is None or meta is None:
        model, meta = load_artifacts()

    X = prepare_data(X_input, meta)
    predict_pool = Pool(X, cat_features=meta["cat_cols"])
    predict = model.predict_proba(predict_pool)[:, 1]
    pos = meta["positive_label"]
    neg = meta["negative_label"]
    answer = np.where(predict >= meta["threshold"], pos, neg)
    return answer, predict


def main():
    pass


if __name__ == "__main__":
    main()
