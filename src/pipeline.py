from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

def load_datasets():
    """Load the four project datasets."""

    datasets = {
        "sales_daily": pd.read_csv(
            RAW_DATA_DIR / "sales_daily.csv",
            parse_dates=["date"]
        ),
        "sku_master": pd.read_csv(
            RAW_DATA_DIR / "sku_master.csv"
        ),
        "calendar": pd.read_csv(
            RAW_DATA_DIR / "calendar.csv",
            parse_dates=["date"]
        ),
        "inventory_snapshots": pd.read_csv(
            RAW_DATA_DIR / "inventory_snapshots.csv",
            parse_dates=["snapshot_date"]
        ),
    }

    return datasets


# --------------------------------------------------
# VALIDATE AND CLEAN DATA
# --------------------------------------------------

def clean_datasets(datasets):
    """Validate required columns and clean records."""

    required_columns = {
        "sales_daily": [
            "date", "sku_id", "units_sold", "unit_price"
        ],
        "sku_master": [
            "sku_id", "product_name", "category",
            "unit_price", "unit_cost", "lead_time_days"
        ],
        "calendar": [
            "date", "day_of_week", "month",
            "quarter", "is_weekend"
        ],
        "inventory_snapshots": [
            "snapshot_date", "sku_id",
            "on_hand_units", "on_order_units"
        ],
    }

    cleaned = {}

    for name, df in datasets.items():

        print(f"\nChecking {name}...")

        # Check required columns
        missing_columns = (
            set(required_columns[name]) - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        original_rows = len(df)

        # Remove exact duplicate records
        df = df.drop_duplicates().copy()

        # Remove records missing essential identifiers
        if name == "sales_daily":
            df = df.dropna(
                subset=["date", "sku_id", "units_sold"]
            )

            df = df[
                (df["units_sold"] >= 0)
                & (df["unit_price"] > 0)
            ]

        elif name == "sku_master":
            df = df.dropna(
                subset=[
                    "sku_id", "product_name", "category",
                    "unit_price", "unit_cost", "lead_time_days"
                ]
            )

            df = df[
                (df["unit_price"] > 0)
                & (df["unit_cost"] >= 0)
                & (df["lead_time_days"] > 0)
            ]

            if df["sku_id"].duplicated().any():
                raise ValueError(
                    "sku_master contains duplicate SKU identifiers."
                )

        elif name == "calendar":
            df = df.dropna(subset=["date"])

            if df["date"].duplicated().any():
                raise ValueError(
                    "calendar contains duplicate dates."
                )

        elif name == "inventory_snapshots":
            df = df.dropna(
                subset=[
                    "snapshot_date", "sku_id",
                    "on_hand_units", "on_order_units"
                ]
            )

            df = df[
                (df["on_hand_units"] >= 0)
                & (df["on_order_units"] >= 0)
            ]

        # Report the results
        removed_rows = original_rows - len(df)

        print(f"Rows before cleaning: {original_rows:,}")
        print(f"Rows after cleaning:  {len(df):,}")
        print(f"Rows removed:         {removed_rows:,}")
        print(f"Missing values:       {df.isna().sum().sum():,}")

        cleaned[name] = df

    # Check relationships between datasets
    valid_skus = set(cleaned["sku_master"]["sku_id"])

    for name in ["sales_daily", "inventory_snapshots"]:
        unknown_skus = (
            set(cleaned[name]["sku_id"]) - valid_skus
        )

        if unknown_skus:
            raise ValueError(
                f"{name} contains unknown SKU identifiers: "
                f"{sorted(unknown_skus)[:10]}"
            )

    return cleaned


# --------------------------------------------------
# SAVE CLEANED DATA
# --------------------------------------------------

def save_datasets(datasets):
    """Save the cleaned datasets as CSV files."""

    for name, df in datasets.items():

        output_path = PROCESSED_DATA_DIR / f"{name}.csv"

        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path.name}")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    print("=" * 55)
    print("PROJECT FORESIGHT - DATA PIPELINE")
    print("=" * 55)

    datasets = load_datasets()

    cleaned_datasets = clean_datasets(datasets)

    save_datasets(cleaned_datasets)

    print("\nData pipeline completed successfully!")


if __name__ == "__main__":
    main()