from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_HORIZON = 6


# --------------------------------------------------
# METRIC
# --------------------------------------------------

def calculate_wape(actual, predicted):
    """Calculate Weighted Absolute Percentage Error."""

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return (
        np.abs(actual - predicted).sum()
        / denominator
    ) * 100


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

def load_weekly_sales():
    sales = pd.read_csv(
        DATA_DIR / "sales_daily.csv",
        parse_dates=["date"]
    )

    # Aggregate daily sales into weekly SKU-level demand.
    sales["week"] = (
        sales["date"]
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    weekly_sales = (
        sales.groupby(
            ["sku_id", "week"],
            as_index=False
        )["units_sold"]
        .sum()
        .rename(columns={"units_sold": "demand"})
    )

    # Include weeks with zero sales for each SKU.
    all_weeks = pd.date_range(
        weekly_sales["week"].min(),
        weekly_sales["week"].max(),
        freq="7D"
    )

    all_skus = weekly_sales["sku_id"].unique()

    complete_index = pd.MultiIndex.from_product(
        [all_skus, all_weeks],
        names=["sku_id", "week"]
    )

    weekly_sales = (
        weekly_sales
        .set_index(["sku_id", "week"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    return weekly_sales.sort_values(["sku_id", "week"])


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def create_features(weekly_sales):
    df = weekly_sales.copy()

    # Historical demand
    df["lag_1"] = (
        df.groupby("sku_id")["demand"].shift(1)
    )

    df["lag_2"] = (
        df.groupby("sku_id")["demand"].shift(2)
    )

    df["lag_4"] = (
        df.groupby("sku_id")["demand"].shift(4)
    )

    # Rolling averages based only on earlier weeks
    df["rolling_mean_4"] = (
        df.groupby("sku_id")["demand"]
        .transform(
            lambda series:
            series.shift(1).rolling(4).mean()
        )
    )

    df["rolling_mean_8"] = (
        df.groupby("sku_id")["demand"]
        .transform(
            lambda series:
            series.shift(1).rolling(8).mean()
        )
    )

    df["month"] = df["week"].dt.month
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)

    return df


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_4",
    "rolling_mean_8",
    "month",
    "week_of_year",
]


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

def main():
    print("=" * 55)
    print("PROJECT FORESIGHT - DEMAND FORECASTING")
    print("=" * 55)

    weekly_sales = load_weekly_sales()
    feature_data = create_features(weekly_sales)

    # Hold out the final six weeks for evaluation.
    all_dates = sorted(feature_data["week"].unique())

    if len(all_dates) < 20:
        raise ValueError(
            "Insufficient weekly history for reliable evaluation."
        )

    test_dates = all_dates[-FORECAST_HORIZON:]
    train_end = test_dates[0]

    train = feature_data[
        (feature_data["week"] < train_end)
        & feature_data[FEATURES].notna().all(axis=1)
    ].copy()

    test = feature_data[
        feature_data["week"].isin(test_dates)
        & feature_data[FEATURES].notna().all(axis=1)
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    X_train = train[FEATURES]
    y_train = train["demand"]

    X_test = test[FEATURES]
    y_test = test["demand"]

    # Baseline: previous week's demand
    baseline_predictions = test["lag_1"].to_numpy()

    # Machine-learning model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    # Evaluation
    baseline_wape = calculate_wape(
        y_test,
        baseline_predictions
    )

    model_wape = calculate_wape(
        y_test,
        predictions
    )

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    print(f"\nTraining records: {len(train):,}")
    print(f"Testing records:  {len(test):,}")

    print(f"\nBaseline WAPE: {baseline_wape:.2f}%")
    print(f"Random Forest WAPE: {model_wape:.2f}%")
    print(f"Random Forest MAE: {model_mae:.2f}")

    # Save evaluation predictions
    results = test[
        ["sku_id", "week", "demand"]
    ].copy()

    results["baseline_prediction"] = baseline_predictions
    results["model_prediction"] = predictions

    results.to_csv(
        REPORT_DIR / "forecast_evaluation.csv",
        index=False
    )

    # Save model
    import joblib

    joblib.dump(
        model,
        MODEL_DIR / "demand_forecast_model.joblib"
    )

    # Save metrics
    metrics = pd.DataFrame([
        {
            "model": "Previous Week Baseline",
            "wape_percent": baseline_wape,
        },
        {
            "model": "Random Forest",
            "wape_percent": model_wape,
        },
    ])

    metrics.to_csv(
        REPORT_DIR / "forecast_metrics.csv",
        index=False
    )

    print("\nForecast evaluation saved.")
    print("Model saved successfully.")
    print("\nForecasting evaluation completed.")


if __name__ == "__main__":
    main()