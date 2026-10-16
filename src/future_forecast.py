from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_4",
    "rolling_mean_8",
    "month",
    "week_of_year",
]


def generate_forecasts(horizon=6):

    sales = pd.read_csv(
        DATA_DIR / "sales_daily.csv",
        parse_dates=["date"],
    )

    model = joblib.load(
        MODEL_DIR / "demand_forecast_model.joblib"
    )

    sales["week"] = (
        sales["date"]
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    weekly = (
        sales.groupby(["sku_id", "week"], as_index=False)
        ["units_sold"]
        .sum()
        .rename(columns={"units_sold": "demand"})
    )

    all_weeks = pd.date_range(
        weekly["week"].min(),
        weekly["week"].max(),
        freq="7D",
    )

    skus = sorted(weekly["sku_id"].unique())

    index = pd.MultiIndex.from_product(
        [skus, all_weeks],
        names=["sku_id", "week"],
    )

    weekly = (
        weekly.set_index(["sku_id", "week"])
        .reindex(index, fill_value=0)
        .reset_index()
        .sort_values(["sku_id", "week"])
    )

    forecast_rows = []

    for sku_id, group in weekly.groupby("sku_id"):

        history = list(group["demand"].astype(float))
        last_week = group["week"].max()

        for step in range(1, horizon + 1):

            next_week = (
                last_week + pd.Timedelta(days=7 * step)
            )

            month = next_week.month
            week_of_year = int(
                next_week.isocalendar().week
            )

            features = pd.DataFrame([{
                "lag_1": history[-1],
                "lag_2": history[-2],
                "lag_4": history[-4],
                "rolling_mean_4": np.mean(history[-4:]),
                "rolling_mean_8": np.mean(history[-8:]),
                "month": month,
                "week_of_year": week_of_year,
            }])[FEATURES]

            prediction = float(
                np.maximum(model.predict(features)[0], 0)
            )

            forecast_rows.append({
                "sku_id": sku_id,
                "week": next_week,
                "predicted_demand": round(prediction, 2),
            })

            # Use the prediction to forecast the next week.
            history.append(prediction)

    result = pd.DataFrame(forecast_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        REPORT_DIR / "future_demand_forecast.csv",
        index=False,
    )

    return result


if __name__ == "__main__":

    forecast = generate_forecasts()

    print("Future demand forecast generated.")
    print(f"Products: {forecast['sku_id'].nunique()}")
    print(f"Forecast records: {len(forecast)}")
    print(f"Saved to: {REPORT_DIR / 'future_demand_forecast.csv'}")

    print("\nSample predictions:")
    print(forecast.head(10).to_string(index=False))