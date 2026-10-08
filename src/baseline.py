import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from features import FEATURES, add_features, clean, flag_outliers

# each fold trains on the past and validates on the next two months,
# which is the same situation as predicting Nov-Dec from Jan-Oct
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

FOLDS = [
    ("train Jan-Jun, validate Jul-Aug", "2025-07-01", "2025-09-01"),
    ("train Jan-Aug, validate Sep-Oct", "2025-09-01", "2025-11-01"),
]


def score(y_true, y_pred):
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    typical = ape < 0.3  # ignore the noisy rows when judging the model itself
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MedAPE%": np.median(ape) * 100,
        "MAE (typical rows)": np.abs(err)[typical].mean(),
        "MAPE% (typical rows)": ape[typical].mean() * 100,
    }


def run_baselines(train_df, valid_df):
    y_valid = valid_df["posted_rate"].values
    results = {}

    # 1. one rate per mile for everything
    rpm = (train_df["posted_rate"] / train_df["distance"]).median()
    results["global rate/mile"] = rpm * valid_df["distance"].values

    # 2. rate per mile by equipment type
    rpm_eq = (train_df["posted_rate"] / train_df["distance"]).groupby(train_df["equipment"]).median()
    results["equipment rate/mile"] = valid_df["equipment"].map(rpm_eq).values * valid_df["distance"].values

    # the models below ignore the noisy rows while fitting
    clean_train = train_df[~flag_outliers(train_df, FEATURES)]
    y_train = np.log(clean_train["posted_rate"])

    # 3. plain linear regression on log rate
    lin = LinearRegression().fit(clean_train[FEATURES], y_train)
    lin_pred = lin.predict(valid_df[FEATURES])
    results["linear (log target)"] = np.exp(lin_pred)

    # 4. gradient boosting on log rate
    gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=15,
                                        min_samples_leaf=40, random_state=42)
    gbm.fit(clean_train[FEATURES], y_train)
    gbm_pred = gbm.predict(valid_df[FEATURES])
    results["gradient boosting (log target)"] = np.exp(gbm_pred)

    # 5. average of the two in log space
    results["linear + boosting average"] = np.exp((lin_pred + gbm_pred) / 2)

    rows = {name: score(y_valid, pred) for name, pred in results.items()}
    return pd.DataFrame(rows).T.round(2)


if __name__ == "__main__":
    data = pd.read_csv("data/train_test.csv")
    weight_fill = data["weight"].abs().median()
    data = add_features(clean(data, weight_fill))

    for name, valid_start, valid_end in FOLDS:
        train_df = data[data["date"] < valid_start]
        valid_df = data[(data["date"] >= valid_start) & (data["date"] < valid_end)]
        print(f"\n{name}  ({len(train_df)} train rows, {len(valid_df)} validation rows)")
        print(run_baselines(train_df, valid_df))