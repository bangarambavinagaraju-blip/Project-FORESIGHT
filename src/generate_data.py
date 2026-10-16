from pathlib import Path

import numpy as np
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Reproducible synthetic data
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def generate_synthetic_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. Product master
    # -------------------------------------------------
    number_of_skus = 30

    categories = [
        "Furniture",
        "Lighting",
        "Kitchen",
        "Decor",
        "Storage",
    ]

    sku_master = pd.DataFrame({
        "sku_id": [
            f"SKU{i:03d}" for i in range(1, number_of_skus + 1)
        ],
        "product_name": [
            f"Home Product {i:03d}"
            for i in range(1, number_of_skus + 1)
        ],
        "category": [
            categories[(i - 1) % len(categories)]
            for i in range(1, number_of_skus + 1)
        ],
        "unit_price": np.round(
            rng.uniform(250, 5000, number_of_skus), 2
        ),
        "unit_cost": np.round(
            rng.uniform(100, 2500, number_of_skus), 2
        ),
        "lead_time_days": rng.integers(
            3, 15, number_of_skus
        ),
    })

    # Ensure costs are lower than selling prices
    sku_master["unit_cost"] = np.minimum(
        sku_master["unit_cost"],
        sku_master["unit_price"] * 0.75
    ).round(2)

    sku_master.to_csv(
        RAW_DATA_DIR / "sku_master.csv", index=False
    )

    # -------------------------------------------------
    # 2. Calendar
    # -------------------------------------------------
    dates = pd.date_range(
        start="2025-01-01",
        end="2026-08-31",
        freq="D"
    )

    calendar = pd.DataFrame({"date": dates})
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["month"] = calendar["date"].dt.month
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["is_weekend"] = (
        calendar["day_of_week"] >= 5
    ).astype(int)

    calendar.to_csv(
        RAW_DATA_DIR / "calendar.csv", index=False
    )

    # -------------------------------------------------
    # 3. Daily sales
    # -------------------------------------------------
    sales_frames = []

    for _, product in sku_master.iterrows():
        base_demand = rng.uniform(3, 15)

        for date in dates:
            weekend_factor = (
                1.20 if date.dayofweek >= 5 else 1.0
            )

            seasonal_factor = (
                1.30 if date.month in [10, 11, 12] else 1.0
            )

            expected_sales = (
                base_demand
                * weekend_factor
                * seasonal_factor
            )

            units_sold = int(
                rng.poisson(expected_sales)
            )

            sales_frames.append({
                "date": date,
                "sku_id": product["sku_id"],
                "units_sold": units_sold,
                "unit_price": product["unit_price"],
            })

    sales_daily = pd.DataFrame(sales_frames)

    sales_daily.to_csv(
        RAW_DATA_DIR / "sales_daily.csv", index=False
    )

    # -------------------------------------------------
    # 4. Inventory snapshots
    # -------------------------------------------------
    inventory_frames = []

    for _, product in sku_master.iterrows():
        inventory_frames.append({
            "snapshot_date": dates[-1],
            "sku_id": product["sku_id"],
            "on_hand_units": int(rng.integers(0, 300)),
            "on_order_units": int(rng.integers(0, 100)),
        })

    inventory_snapshots = pd.DataFrame(inventory_frames)

    inventory_snapshots.to_csv(
        RAW_DATA_DIR / "inventory_snapshots.csv",
        index=False
    )

    print("Synthetic datasets generated successfully.")
    print(f"Location: {RAW_DATA_DIR}")
    print("\nDataset sizes:")

    for filename in [
        "sales_daily.csv",
        "sku_master.csv",
        "calendar.csv",
        "inventory_snapshots.csv",
    ]:
        df = pd.read_csv(RAW_DATA_DIR / filename)
        print(f"{filename}: {len(df):,} rows")


if __name__ == "__main__":
    generate_synthetic_data()