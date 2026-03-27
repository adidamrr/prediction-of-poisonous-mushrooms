import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def drop_objects(X_input, y_input, col_name, max_t=None, min_t=None):
    X = X_input.copy()
    y = y_input.copy()

    if max_t is not None:
        y = y.drop(y[X[col_name] > max_t].index)
        X = X.drop(X[X[col_name] > max_t].index)

    if min_t is not None:
        y = y.drop(y[X[col_name] < min_t].index)
        X = X.drop(X[X[col_name] < min_t].index)

    return X, y


def find_best_t(y_va, p_va, target=0.5, cnt_of_variance=20) -> float:
    t_list = np.linspace(0, 1, cnt_of_variance)
    res_list = []
    for t in t_list:
        pred_va = (p_va >= t).astype(int)
        val_recall = recall_score(y_va, pred_va, zero_division=0)
        val_precision = precision_score(y_va, pred_va, zero_division=0)
        res_list.append(
            {
                "t": t,
                "val_recall": val_recall,
                "val_precision": val_precision,
            }
        )

    best_res_list = [res for res in res_list if res["val_recall"] >= target]
    if best_res_list:
        best_res = sorted(
            best_res_list, key=lambda x: (x["val_precision"], x["val_recall"])
        )[-1]
        best_t = best_res["t"]
        print(
            f"best t = {best_t} with recall = {best_res['val_recall']} "
            f"and precision = {best_res['val_precision']}"
        )
        return float(best_t)

    print("Нет такого значения recall")
    return 0.5


def main():
    train_path = "data/train.csv"
    model_path = "model/model.cbm"
    meta_path = "model/meta.json"

    os.makedirs("model", exist_ok=True)

    df = pd.read_csv(train_path, index_col="id")

    num_features = ["cap-diameter", "stem-height", "stem-width"]
    cat_features = [
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
    feature_cols = cat_features + num_features
    allowed_cols = set(feature_cols + ["class"])
    unused_cols = [col for col in df.columns if col not in allowed_cols]
    df = df.drop(columns=unused_cols)

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_cols],
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    y_tr = (y_train.astype(str) == "p").astype(int)
    y_va = (y_test.astype(str) == "p").astype(int)

    cat_cols = cat_features
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    nan_token = "__nan__"
    X_tr = X_train.copy()
    X_va = X_test.copy()
    X_tr[num_features] = np.log1p(X_tr[num_features])
    X_va[num_features] = np.log1p(X_va[num_features])
    X_tr[cat_cols] = X_tr[cat_cols].fillna(nan_token).astype(str)
    X_va[cat_cols] = X_va[cat_cols].fillna(nan_token).astype(str)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_va, y_va, cat_features=cat_idx)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=100,
        learning_rate=0.08,
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
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    p_va = model.predict_proba(val_pool)[:, 1]
    print("VAL AUC:", roc_auc_score(y_va, p_va))

    best_t = find_best_t(y_va, p_va, target=0.9, cnt_of_variance=40)

    X, y = drop_objects(X, y, "cap-diameter", max_t=57)
    X, y = drop_objects(X, y, "stem-height", max_t=57)
    X, y = drop_objects(X, y, "stem-width", max_t=60)

    y = (y.astype(str) == "p").astype(int)

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    X_copy = X.copy()
    X_copy[num_features] = np.log1p(X_copy[num_features])
    X_copy[cat_cols] = X_copy[cat_cols].fillna(nan_token).astype(str)

    full_pool = Pool(X_copy, y, cat_features=cat_idx)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=100,
        learning_rate=0.05,
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
    model.fit(full_pool)

    model.save_model(model_path)

    meta = {
        "threshold": float(best_t),
        "num_cols": num_features,
        "cat_cols": cat_cols,
        "nan_token": nan_token,
        "positive_label": "p",
        "negative_label": "e",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
