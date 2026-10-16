from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 50)
    print("PROJECT FORESIGHT - EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # Load processed datasets
    sales = pd.read_csv(
        DATA_DIR / "sales_daily.csv",
        parse_dates=["date"]
    )

    products = pd.read_csv(
        DATA_DIR / "sku_master.csv"
    )

    # Merge sales with product information
    df = sales.merge(
        products[["sku_id", "product_name", "category"]],
        on="sku_id",
        how="left",
        validate="many_to_one"
    )

    # Overall summary
    print("\n1. OVERALL SALES SUMMARY")

    print(f"Total sales records: {len(df):,}")
    print(f"Unique products: {df['sku_id'].nunique()}")
    print(f"Total units sold: {df['units_sold'].sum():,}")
    print(
        f"Total recorded sales value: "
        f"Rs. {(df['units_sold'] * df['unit_price']).sum():,.2f}"
    )

    # Daily sales
    daily_sales = (
        df.groupby("date", as_index=False)["units_sold"]
        .sum()
        .sort_values("date")
    )

    daily_sales.to_csv(
        REPORT_DIR / "daily_sales_summary.csv",
        index=False
    )

    # Monthly sales
    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly_sales = (
        df.groupby("month", as_index=False)["units_sold"]
        .sum()
    )

    monthly_sales.to_csv(
        REPORT_DIR / "monthly_sales_summary.csv",
        index=False
    )

    print("\n2. MONTHLY SALES")
    print(monthly_sales.to_string(index=False))

    # Top-selling products
    top_products = (
        df.groupby("sku_id", as_index=False)["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
        .head(10)
    )

    top_products = top_products.merge(
        products[["sku_id", "product_name", "category"]],
        on="sku_id",
        how="left"
    )

    top_products.to_csv(
        REPORT_DIR / "top_products.csv",
        index=False
    )

    print("\n3. TOP 10 PRODUCTS")
    print(top_products.to_string(index=False))

    # Category analysis
    category_sales = (
        df.groupby("category", as_index=False)["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
    )

    category_sales.to_csv(
        REPORT_DIR / "category_sales.csv",
        index=False
    )

    print("\n4. SALES BY CATEGORY")
    print(category_sales.to_string(index=False))

    # Plot daily sales
    plt.figure(figsize=(12, 5))
    plt.plot(
        daily_sales["date"],
        daily_sales["units_sold"]
    )
    plt.title("Daily Units Sold - Project FORESIGHT")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = REPORT_DIR / "daily_sales_trend.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # Plot category sales
    plt.figure(figsize=(9, 5))
    plt.bar(
        category_sales["category"],
        category_sales["units_sold"]
    )
    plt.title("Units Sold by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=30)
    plt.tight_layout()

    category_chart_path = REPORT_DIR / "category_sales.png"
    plt.savefig(category_chart_path, dpi=150)
    plt.close()

    print("\n5. REPORTS GENERATED")
    print(f"Daily sales chart: {chart_path}")
    print(f"Category chart: {category_chart_path}")
    print(f"Reports folder: {REPORT_DIR}")

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    main()