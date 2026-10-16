from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT FORESIGHT
# AI-Powered Demand & Inventory Intelligence Platform
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    sales = pd.read_csv(
        DATA_DIR / "sales_daily.csv",
        parse_dates=["date"],
    )

    products = pd.read_csv(
        DATA_DIR / "sku_master.csv",
    )

    inventory_risk = pd.read_csv(
        REPORT_DIR / "inventory_risk_report.csv",
    )

    forecast = pd.read_csv(
        REPORT_DIR / "future_demand_forecast.csv",
        parse_dates=["week"],
    )

    metrics_path = REPORT_DIR / "forecast_metrics.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.DataFrame()

    return sales, products, inventory_risk, forecast, metrics


try:
    (
        sales,
        products,
        inventory_risk,
        forecast,
        forecast_metrics,
    ) = load_data()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info(
        "Check that the data pipeline, risk scoring, and "
        "future forecasting scripts have completed successfully."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📦 Project FORESIGHT")

st.subheader(
    "AI-Powered Demand & Inventory Intelligence Platform"
)

st.caption(
    "Development dashboard using synthetic data. "
    "Forecasts and financial estimates are illustrative, "
    "not verified business results."
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Filters")

categories = sorted(
    products["category"].dropna().unique().tolist()
)

selected_categories = st.sidebar.multiselect(
    "Select product categories",
    options=categories,
    default=categories,
)

filtered_products = products[
    products["category"].isin(selected_categories)
].copy()

selected_skus = filtered_products["sku_id"].tolist()

filtered_sales = sales[
    sales["sku_id"].isin(selected_skus)
].copy()

filtered_risk = inventory_risk[
    inventory_risk["sku_id"].isin(selected_skus)
].copy()

filtered_forecast = forecast[
    forecast["sku_id"].isin(selected_skus)
].copy()


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

total_products = filtered_products["sku_id"].nunique()

total_units = int(filtered_sales["units_sold"].sum())

reorder_count = int(
    (
        filtered_risk["risk_category"] == "Reorder Now"
    ).sum()
)

stockout_exposure = filtered_risk[
    "estimated_stockout_exposure_rs"
].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Products",
    f"{total_products:,}",
)

col2.metric(
    "Units Sold",
    f"{total_units:,}",
)

col3.metric(
    "Products to Reorder",
    f"{reorder_count:,}",
)

col4.metric(
    "Estimated Stockout Exposure",
    f"₹{stockout_exposure:,.0f}",
)

st.divider()


# ============================================================
# DAILY SALES TREND
# ============================================================

st.header("📈 Daily Sales Trend")

daily_sales = (
    filtered_sales.groupby("date", as_index=False)["units_sold"]
    .sum()
    .sort_values("date")
)

if not daily_sales.empty:
    daily_chart = px.line(
        daily_sales,
        x="date",
        y="units_sold",
        title="Daily Units Sold",
    )

    daily_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
    )

    st.plotly_chart(
        daily_chart,
        width="stretch"
    )
else:
    st.info("No sales data is available for the selected categories.")


# ============================================================
# CATEGORY SALES AND INVENTORY RISK
# ============================================================

left, right = st.columns(2)

with left:
    st.subheader("Sales by Category")

    category_sales = filtered_sales.merge(
        filtered_products[["sku_id", "category"]],
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    category_summary = (
        category_sales.groupby(
            "category",
            as_index=False,
        )["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
    )

    if not category_summary.empty:
        category_chart = px.bar(
            category_summary,
            x="category",
            y="units_sold",
            title="Units Sold by Category",
        )

        category_chart.update_layout(
            xaxis_title="Category",
            yaxis_title="Units Sold",
        )

        st.plotly_chart(
            category_chart,
            width="stretch",
        )
    else:
        st.info("No category sales data available.")


with right:
    st.subheader("Inventory Risk Distribution")

    if not filtered_risk.empty:
        risk_summary = (
            filtered_risk["risk_category"]
            .value_counts()
            .rename_axis("risk_category")
            .reset_index(name="products")
        )

        risk_chart = px.pie(
            risk_summary,
            names="risk_category",
            values="products",
            title="Products by Risk Category",
            hole=0.4,
        )

        st.plotly_chart(
            risk_chart,
            width="stretch",
        )
    else:
        st.info("No inventory risk data available.")


st.divider()


# ============================================================
# INVENTORY RECOMMENDATIONS
# ============================================================

st.header("🚨 Inventory Recommendations")

if not filtered_risk.empty:

    risk_options = sorted(
        filtered_risk["risk_category"].dropna().unique().tolist()
    )

    selected_risks = st.multiselect(
        "Filter by risk category",
        options=risk_options,
        default=risk_options,
    )

    recommendations = filtered_risk[
        filtered_risk["risk_category"].isin(selected_risks)
    ].copy()

    display_columns = [
        "sku_id",
        "product_name",
        "category",
        "on_hand_units",
        "on_order_units",
        "average_daily_demand",
        "risk_category",
        "recommended_action",
        "estimated_stockout_exposure_rs",
        "estimated_excess_stock_value_rs",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in recommendations.columns
    ]

    st.dataframe(
        recommendations[available_columns],
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Inventory Risk Report",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name="inventory_risk_report.csv",
        mime="text/csv",
    )

else:
    st.info("No inventory recommendations are available.")


st.divider()


# ============================================================
# TOP-SELLING PRODUCTS
# ============================================================

st.header("🏆 Top-Selling Products")

if not filtered_sales.empty:

    top_products = (
        filtered_sales.groupby(
            "sku_id",
            as_index=False,
        )["units_sold"]
        .sum()
        .merge(
            filtered_products[
                ["sku_id", "product_name", "category"]
            ],
            on="sku_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values("units_sold", ascending=False)
        .head(10)
    )

    st.dataframe(
        top_products,
        width="stretch",
        hide_index=True,
    )

else:
    st.info("No top-selling products are available.")


st.divider()


# ============================================================
# SIX-WEEK DEMAND FORECAST
# ============================================================

st.header("🔮 Six-Week Demand Forecast")

st.caption(
    "Predictions generated by the Random Forest model "
    "using synthetic development data."
)

if not filtered_forecast.empty:

    forecast_skus = sorted(
        filtered_forecast["sku_id"].unique().tolist()
    )

    selected_sku = st.selectbox(
        "Select a product to forecast",
        options=forecast_skus,
    )

    sku_forecast = (
        filtered_forecast[
            filtered_forecast["sku_id"] == selected_sku
        ]
        .sort_values("week")
        .copy()
    )

    forecast_chart = px.line(
        sku_forecast,
        x="week",
        y="predicted_demand",
        markers=True,
        title=f"Predicted Weekly Demand — {selected_sku}",
    )

    forecast_chart.update_layout(
        xaxis_title="Forecast Week",
        yaxis_title="Predicted Units",
    )

    st.plotly_chart(
        forecast_chart,
        width="stretch",
    )

    st.subheader("Forecast Details")

    st.dataframe(
        sku_forecast,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Forecast CSV",
        data=sku_forecast.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_sku}_forecast.csv",
        mime="text/csv",
    )

else:
    st.warning(
        "No forecast data is available for the selected categories."
    )


# ============================================================
# FORECAST MODEL EVALUATION
# ============================================================

st.divider()

st.header("📊 Forecast Model Evaluation")

if not forecast_metrics.empty:

    if {"model", "wape_percent"}.issubset(
        forecast_metrics.columns
    ):

        st.caption(
            "WAPE values below describe the existing historical "
            "holdout evaluation. They are not future forecast accuracy."
        )

        st.dataframe(
            forecast_metrics,
            width="stretch",
            hide_index=True,
        )

        evaluation_chart = px.bar(
            forecast_metrics,
            x="model",
            y="wape_percent",
            title="Historical Holdout WAPE Comparison",
            text="wape_percent",
        )

        evaluation_chart.update_layout(
            xaxis_title="Model",
            yaxis_title="WAPE (%)",
        )

        st.plotly_chart(
            evaluation_chart,
            width="stretch",
        )

    else:
        st.warning(
            "The forecast metrics file does not contain the "
            "expected model and wape_percent columns."
        )

else:
    st.info("Forecast evaluation metrics are not available.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Project FORESIGHT | Synthetic development dataset | "
    "Forecasts and financial estimates require further validation."
)
