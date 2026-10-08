import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from baseline import score
from features import FEATURES, add_features, clean, flag_outliers

# three rolling folds, each validating on the two months after the training window
FOLDS = [
    ("train Jan-Apr, validate May-Jun", "2025-05-01", "2025-07-01"),
    ("train Jan-Jun, validate Jul-Aug", "2025-07-01", "2025-09-01"),
    ("train Jan-Aug, validate Sep-Oct", "2025-09-01", "2025-11-01"),
]

# small grid, the data is mostly smooth so we don't need much
GBM_GRID = {
    "loss": ["squared_error", "absolute_error"],
    "max_leaf_nodes": [8, 15, 31],
    "min_samples_leaf": [40, 100],
}
LEARNING_RATE = 0.04
MAX_ITER = 500
BLEND_WEIGHTS = [0.3, 0.5, 0.7]  # share given to the boosting model


def fit_gbm(x, y, **params):
    model = HistGradientBoostingRegressor(learning_rate=LEARNING_RATE, max_iter=MAX_ITER,
                                          random_state=42, **params)
    return model.fit(x, y)


def split_fold(data, valid_start, valid_end):
    train_df = data[data["date"] < valid_start]
    valid_df = data[(data["date"] >= valid_start) & (data["date"] < valid_end)]
    # noisy rows are dropped from training only, never from validation
    train_df = train_df[~flag_outliers(train_df, FEATURES)]
    return train_df, valid_df


def tune(data):
    """Try each boosting setting and blend weight on all folds, return a results table."""
    rows = []
    keys = list(GBM_GRID)
    for values in itertools.product(*GBM_GRID.values()):
        params = dict(zip(keys, values))
        fold_scores = {w: [] for w in BLEND_WEIGHTS + [1.0]}
        for _, start, end in FOLDS:
            train_df, valid_df = split_fold(data, start, end)
            y_train = np.log(train_df["posted_rate"])
            lin_pred = LinearRegression().fit(train_df[FEATURES], y_train).predict(valid_df[FEATURES])
            gbm_pred = fit_gbm(train_df[FEATURES], y_train, **params).predict(valid_df[FEATURES])
            for w in fold_scores:
                blended = np.exp((1 - w) * lin_pred + w * gbm_pred)
                fold_scores[w].append(score(valid_df["posted_rate"].values, blended))
        for w, scores in fold_scores.items():
            avg = pd.DataFrame(scores).mean()
            rows.append({**params, "boost_weight": w,
                         "MAE": avg["MAE"], "MedAPE%": avg["MedAPE%"],
                         "typical MAE": avg["MAE (typical rows)"],
                         "typical MAPE%": avg["MAPE% (typical rows)"]})
    return pd.DataFrame(rows).sort_values("typical MAE").reset_index(drop=True)


def fit_final(data, params, boost_weight):
    train_df = data[~flag_outliers(data, FEATURES)]
    y_train = np.log(train_df["posted_rate"])
    lin = LinearRegression().fit(train_df[FEATURES], y_train)
    gbm = fit_gbm(train_df[FEATURES], y_train, **params)

    def predict(frame):
        log_pred = (1 - boost_weight) * lin.predict(frame[FEATURES]) + boost_weight * gbm.predict(frame[FEATURES])
        return np.exp(log_pred)

    return predict


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    train_raw = pd.read_csv("data/train_test.csv")
    weight_fill = train_raw["weight"].abs().median()
    train_data = add_features(clean(train_raw, weight_fill))

    results = tune(train_data)
    results.round(3).to_csv("outputs/tuning_results.csv", index=False)
    print("Top 8 settings (averaged over the three folds):")
    print(results.head(8).round(3))

    best = results.iloc[0]
    best_params = {
        "loss": best["loss"],
        "max_leaf_nodes": int(best["max_leaf_nodes"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }
    print("\nChosen:", best_params, "boosting weight", best["boost_weight"])

    predict = fit_final(train_data, best_params, best["boost_weight"])

    # final predictions for validation.csv
    valid_raw = pd.read_csv("data/validation.csv")
    valid_data = add_features(clean(valid_raw, weight_fill))
    template = pd.read_csv("data/validation_predictions_template.csv")
    predictions = pd.DataFrame({"load_id": valid_data["load_id"],
                                "predicted_rate": predict(valid_data).round(2)})
    predictions = template[["load_id"]].merge(predictions, on="load_id", how="left")
    assert predictions["predicted_rate"].notna().all() and len(predictions) == 12_000
    predictions.to_csv("validation_predictions.csv", index=False)
    print("\nSaved validation_predictions.csv")
    print(predictions["predicted_rate"].describe().round(1))