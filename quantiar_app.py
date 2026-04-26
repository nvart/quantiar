import json
import os
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


st.set_page_config(page_title="QuantiAR", layout="wide")

st.title("📊 QuantiAR - Portfolio Analytics")
st.caption("Backtesting de carteras con comparación vs Dólar MEP e inflación argentina.")

END_DATE = datetime.today().strftime("%Y-%m-%d")
MIN_DATA_DATE = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
DEFAULT_ANALYSIS_DATE = "2020-01-01"
WEIGHT_TOLERANCE = 0.001

AVAILABLE_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "GLD": "GLD",
    "TLT": "TLT",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

PORTFOLIOS_FILE = "portfolios.json"


def load_saved_portfolios():
    if not os.path.exists(PORTFOLIOS_FILE):
        return {}
    try:
        with open(PORTFOLIOS_FILE, "r") as file:
            return json.load(file)
    except Exception:
        return {}


def save_portfolios(portfolios):
    with open(PORTFOLIOS_FILE, "w") as file:
        json.dump(portfolios, file, indent=4)


@st.cache_data
def download_yahoo(ticker, start_date):
    period1 = int(pd.Timestamp(start_date).timestamp())
    period2 = int(pd.Timestamp(END_DATE).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    data = response.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    close = result["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit="s").date,
        ticker: close
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.dropna()

    return df[ticker]


@st.cache_data
def download_prices(selected_assets, data_start_date):
    data = {}

    for asset in selected_assets:
        yahoo_ticker = AVAILABLE_TICKERS[asset]
        data[asset] = download_yahoo(yahoo_ticker, data_start_date)

    prices = pd.DataFrame(data)
    prices = prices.sort_index()
    prices = prices.ffill().dropna()

    return prices


@st.cache_data
def get_mep_historico():
    url = "https://api.argentinadatos.com/v1/cotizaciones/dolares/bolsa"
    response = requests.get(url, timeout=20)
