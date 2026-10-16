\# Project FORESIGHT

\## AI-Powered Demand \& Inventory Intelligence Platform



Project FORESIGHT is a data science project that analyzes product sales, forecasts future demand, and identifies potential inventory risks.



This development version uses synthetic data. Forecasts and financial estimates are illustrative and have not been validated against real business data.



\## Objectives



\- Analyze historical product sales.

\- Clean and prepare datasets.

\- Explore sales trends and product performance.

\- Forecast weekly product demand.

\- Identify potential stockout risks.

\- Recommend inventory actions.

\- Display insights through an interactive dashboard.



\## Technology Stack



\- Python

\- pandas

\- NumPy

\- scikit-learn

\- Matplotlib

\- Plotly

\- Streamlit



\## Project Structure



\- app/dashboard.py — Interactive dashboard

\- data/raw/ — Raw synthetic datasets

\- data/processed/ — Processed datasets

\- models/ — Trained forecasting model

\- reports/ — Analysis and forecasting reports

\- src/eda.py — Exploratory data analysis

\- src/forecast.py — Forecast model evaluation

\- src/future\_forecast.py — Future demand forecasting

\- src/generate\_data.py — Synthetic data generation

\- src/pipeline.py — Data processing pipeline

\- src/risk.py — Inventory risk analysis



\## Current Forecast Evaluation



The current development run reported:



\- Baseline WAPE: 27.31%

\- Random Forest WAPE: 25.82%

\- Random Forest MAE: 16.81



These preliminary results use synthetic data and require further validation.



\## Dashboard Features



\- Daily sales trends

\- Sales by category

\- Top-selling products

\- Inventory risk distribution

\- Inventory recommendations

\- Six-week demand forecasts

\- Product category filters



\## Installation



Create and activate a virtual environment:



&#x20;   python -m venv .venv



On Windows PowerShell:



&#x20;   .\\.venv\\Scripts\\Activate.ps1



Install dependencies:



&#x20;   python -m pip install -r requirements.txt



\## Run the Project



Run the data pipeline:



&#x20;   python src/pipeline.py



Run inventory risk analysis:



&#x20;   python src/risk.py



Generate future forecasts:



&#x20;   python src/future\_forecast.py



Launch the dashboard:



&#x20;   python -m streamlit run app/dashboard.py



Open the local URL shown in the terminal.



\## Limitations



\- The current dataset is synthetic.

\- Financial estimates are illustrative.

\- Inventory recommendations use simplified rules.

\- Forecast performance requires additional validation.

\- Predictions are estimates, not guarantees.



\## Future Improvements



\- Rolling-origin forecast validation

\- Improved inventory recommendations

\- Forecast-versus-actual monitoring

\- Automated model retraining

\- Public dashboard deployment



\## Author



Nagaraju Bangarambavi



Project FORESIGHT — Demand \& Inventory Intelligence

