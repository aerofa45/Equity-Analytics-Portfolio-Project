# Equity Portfolio Analytics – End-to-End Job Portfolio Project

This project demonstrates a full **SQL + Python + Dashboard** workflow suitable for a data analyst role.


## Components

1. **SQLite Database (`portfolio.db`)**
   - Table `prices` with daily adjusted close prices.
   - Table `portfolio_weights` with target weights per ticker.

2. **Jupyter Notebook (`portfolio_end_to_end.ipynb`)**
   - Downloads price data from Yahoo Finance (via `yfinance`).
   - Loads data into SQLite.
   - Runs SQL queries for:
     - Daily returns
     - Portfolio returns
     - Volatility and correlations
   - Uses pandas & plotly for analysis and charts.

3. **Streamlit App (`portfolio_dashboard.py`)**
   - Interactive dashboard:
     - KPI cards (return, volatility, Sharpe, drawdown)
     - Portfolio vs. benchmark chart
     - Per-asset performance
     - Risk–return scatter
     - Correlation heatmap
   - Reads from the same SQLite database.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `portfolio_end_to_end.ipynb` in Jupyter / VS Code / Colab and run all cells.
This will create `portfolio.db` and populate it with recent market data.

### 3. Launch the dashboard

```bash
streamlit run portfolio_dashboard.py
```

The app will connect to `portfolio.db` and display the analytics dashboard.

[![Open in Streamlit](https://img.shields.io/badge/Live%20App-Streamlit-blue?style=for-the-badge)](https://equity-analytics-portfolio-project-5mou7if7fsgkzixa36e3sx.streamlit.app/)


> "I built an end-to-end portfolio analytics system: SQL for data storage & queries, Python for analytics, and Streamlit for the business-facing dashboard."
