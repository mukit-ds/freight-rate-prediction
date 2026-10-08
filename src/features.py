import numpy as np
import pandas as pd


def clean(df, weight_fill=None):
    """Fix the data issues found during exploration."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # some weights are negative but look like normal values with the sign flipped
    df["weight"] = df["weight"].abs()
    df["weight_missing"] = df["weight"].isna().astype(int)
    if weight_fill is None:
        weight_fill = df["weight"].median()
    df["weight"] = df["weight"].fillna(weight_fill)

    # market_index hardly changes within a day, so fill gaps with that day's median
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    day_median = df.groupby("date")["market_index"].transform("median")
    df["market_index"] = df["market_index"].fillna(day_median)
    # if a whole day is missing, carry over the closest day
    df = df.sort_values("date")
    df["market_index"] = df["market_index"].interpolate(limit_direction="both")
    return df.sort_index()


def add_features(df):
    df = df.copy()
    df["log_distance"] = np.log(df["distance"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_month"] = df["date"].dt.day
    df["equipment_code"] = df["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})

    # how far east-west / north-south the lane runs
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["is_short_haul"] = (df["distance"] < 250).astype(int)
    df["very_short_haul"] = (df["distance"] < 100).astype(int)
    df["heavy_load"] = (df["weight"] > 42_000).astype(int)

    # flatbed rates creep up through the last month of each quarter (Mar, Jun, Sep),
    # so track how far into the quarter we are and let flatbed have its own slope
    quarter = df["date"].dt.to_period("Q")
    quarter_start = quarter.dt.start_time
    quarter_days = (quarter.dt.end_time.dt.normalize() - quarter_start).dt.days
    df["quarter_progress"] = (df["date"] - quarter_start).dt.days / quarter_days
    is_flatbed = (df["equipment"] == "Flatbed").astype(int)
    df["flatbed_progress"] = is_flatbed * df["quarter_progress"]
    df["flatbed_progress_sq"] = is_flatbed * df["quarter_progress"] ** 2

    # very low and very high quote signals both sit above the middle
    df["quote_signal_gap"] = (df["quote_signal"] - 2.05).abs()

    # day-level market context, built from the features only (no rates involved)
    by_day = df.groupby("date").agg(day_quote=("quote_signal", "mean"),
                                    day_market=("market_index", "median"))
    df["day_quote_mean"] = df["date"].map(by_day["day_quote"])
    df["market_7day"] = df["date"].map(
        by_day["day_market"].rolling(7, min_periods=1, center=True).mean())
    return df


def flag_outliers(df, features, threshold=0.3):
    """Mark rows whose rate is far off what a simple log-linear model expects.

    About 1.4% of loads have a rate 2-5x too high or 3-5x too low that nothing in
    the features explains, so we leave them out when fitting.
    """
    from sklearn.linear_model import LinearRegression
    y = np.log(df["posted_rate"])
    model = LinearRegression().fit(df[features], y)
    return (y - model.predict(df[features])).abs() > threshold


FEATURES = [
    "distance", "log_distance", "weight", "equipment_code",
    "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "lat_diff", "lon_diff", "is_short_haul", "very_short_haul", "heavy_load",
    "month", "day_of_week", "day_of_year", "day_of_month",
    "weight_missing", "market_index_missing",
    "quarter_progress", "flatbed_progress", "flatbed_progress_sq",
    "quote_signal_gap", "day_quote_mean", "market_7day",
]