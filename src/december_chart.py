import numpy as np
import pandas as pd

from features import add_features, clean
from train_final import fit_final


def city_coordinates(*frames):
    """Each city has one fixed coordinate pair in the data, so build a lookup."""
    parts = []
    for df in frames:
        parts.append(df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def build_december_features(chart, valid_data, coords):
    """Turn the 31 fixed chart rows into model features.

    The chart file only has the lane, distance, equipment, weight and date, so
    market_index and quote_signal come from the real December loads in validation.csv
    (the daily median). The day-level context features are copied from the same days.
    """
    daily = valid_data.groupby("date").agg(market_index=("market_index", "median"),
                                           quote_signal=("quote_signal", "median"),
                                           day_quote_mean=("day_quote_mean", "first"),
                                           market_7day=("market_7day", "first"))
    out = chart.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["load_id"] = "chart"
    out["posted_rate"] = np.nan
    out["pickup_lat"] = coords.loc[out["pickup"], "lat"].values
    out["pickup_lon"] = coords.loc[out["pickup"], "lon"].values
    out["delivery_lat"] = coords.loc[out["delivery"], "lat"].values
    out["delivery_lon"] = coords.loc[out["delivery"], "lon"].values
    out["weight_missing"] = 0
    out["market_index_missing"] = 0
    out = out.join(daily[["market_index", "quote_signal"]], on="date")

    out = add_features(out)
    # add_features would build day-level features from these 31 rows only,
    # so overwrite them with the values from all of that day's validation loads
    out["day_quote_mean"] = out["date"].map(daily["day_quote_mean"])
    out["market_7day"] = out["date"].map(daily["market_7day"])
    return out


if __name__ == "__main__":
    from features import FEATURES  # noqa: F401  (kept so the feature list is easy to find)

    train_raw = pd.read_csv("data/train_test.csv")
    weight_fill = train_raw["weight"].abs().median()
    train_data = add_features(clean(train_raw, weight_fill))
    valid_raw = pd.read_csv("data/validation.csv")
    valid_data = add_features(clean(valid_raw, weight_fill))

    # use the best setting found during tuning
    best = pd.read_csv("outputs/tuning_results.csv").iloc[0]
    params = {"loss": best["loss"], "max_leaf_nodes": int(best["max_leaf_nodes"]),
              "min_samples_leaf": int(best["min_samples_leaf"])}
    predict = fit_final(train_data, params, best["boost_weight"])

    chart = pd.read_csv("data/december_chart_inputs.csv")
    coords = city_coordinates(train_raw, valid_raw)
    features = build_december_features(chart, valid_data, coords)

    chart["predicted_rate"] = predict(features).round(2)
    chart.to_csv("data/december_chart_inputs.csv", index=False)
    print(chart[["date", "predicted_rate"]].to_string(index=False))
    print("\nMin / max / mean:", chart["predicted_rate"].min(), chart["predicted_rate"].max(),
          round(chart["predicted_rate"].mean(), 2))