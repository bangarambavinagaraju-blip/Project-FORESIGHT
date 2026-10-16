from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# INVENTORY RISK SCORING
# --------------------------------------------------

def main():

    print("=" * 55)
    print("PROJECT FORESIGHT - INVENTORY RISK SCORING")
    print("=" * 55)

    # Load data
    inventory = pd.read_csv(
        DATA_DIR / "inventory_snapshots.csv",
        parse_dates=["snapshot_date"]
    )

    products = pd.read_csv(
        DATA_DIR / "sku_master.csv"
    )

    sales = pd.read_csv(
        DATA_DIR / "sales_daily.csv",
        parse_dates=["date"]
    )

    # Estimate average daily demand using the latest
    # 28 days of available sales history.
    latest_date = sales["date"].max()

    recent_sales = sales[
        sales["date"] > latest_date - pd.Timedelta(days=28)
    ]

    average_demand = (
        recent_sales.groupby("sku_id")["units_sold"]
        .mean()
        .rename("average_daily_demand")
    )

    # Join inventory, product and demand information.
    risk = (
        inventory
        .merge(
            products,
            on="sku_id",
            how="left",
            validate="many_to_one"
        )
        .merge(
            average_demand,
            on="sku_id",
            how="left",
            validate="many_to_one"
        )
    )

    risk["average_daily_demand"] = (
        risk["average_daily_demand"].fillna(0)
    )

    # Stock available, including incoming orders.
    risk["inventory_position"] = (
        risk["on_hand_units"] + risk["on_order_units"]
    )

    # Estimated demand during supplier lead time.
    risk["lead_time_demand"] = (
        risk["average_daily_demand"] * risk["lead_time_days"]
    )

    # Estimated days of inventory on hand.
    risk["days_of_inventory"] = (
        risk["on_hand_units"]
        / risk["average_daily_demand"].replace(0, float("nan"))
    )

    # --------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------

    def classify_risk(row):

        # No recent demand requires manual investigation.
        if row["average_daily_demand"] == 0:
            return "Watch / Volatile"

        # Inventory cannot cover estimated lead-time demand.
        if row["inventory_position"] < row["lead_time_demand"]:
            return "Reorder Now"

        # Excess stock: more than 90 days of expected demand.
        if (
            row["on_hand_units"]
            > row["average_daily_demand"] * 90
        ):
            return "Markdown / Clear"

        return "Healthy"

    risk["risk_category"] = risk.apply(
        classify_risk,
        axis=1
    )

    # --------------------------------------------------
    # RECOMMENDED ACTIONS
    # --------------------------------------------------

    action_map = {
        "Reorder Now": "Review supplier order and replenish stock",
        "Markdown / Clear": "Review markdown or clearance options",
        "Watch / Volatile": "Investigate demand and inventory history",
        "Healthy": "Continue routine inventory monitoring",
    }

    risk["recommended_action"] = (
        risk["risk_category"].map(action_map)
    )

    # --------------------------------------------------
    # ESTIMATED FINANCIAL IMPACT
    # --------------------------------------------------

    risk["estimated_stockout_exposure_rs"] = (
        (
            risk["lead_time_demand"]
            - risk["inventory_position"]
        ).clip(lower=0)
        * risk["unit_price"]
    )

    risk["estimated_excess_stock_value_rs"] = (
        (
            risk["on_hand_units"]
            - risk["average_daily_demand"] * 90
        ).clip(lower=0)
        * risk["unit_cost"]
    )

    # Round numeric outputs.
    numeric_columns = [
        "average_daily_demand",
        "lead_time_demand",
        "days_of_inventory",
        "estimated_stockout_exposure_rs",
        "estimated_excess_stock_value_rs",
    ]

    risk[numeric_columns] = risk[numeric_columns].round(2)

    # Save report.
    output_path = REPORT_DIR / "inventory_risk_report.csv"

    risk.to_csv(output_path, index=False)

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\nInventory risk categories:")

    print(
        risk["risk_category"]
        .value_counts()
        .to_string()
    )

    print("\nEstimated stockout exposure:")
    print(
        f"Rs. {risk['estimated_stockout_exposure_rs'].sum():,.2f}"
    )

    print("\nEstimated excess stock value:")
    print(
        f"Rs. {risk['estimated_excess_stock_value_rs'].sum():,.2f}"
    )

    print(f"\nReport saved: {output_path}")
    print("\nInventory risk scoring completed.")


if __name__ == "__main__":
    main()