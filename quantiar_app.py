import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import plotly.graph_objects as go


st.set_page_config(page_title="QuantiAR", layout="wide")

st.markdown("""
<h1 style='color:#00A3E0; font-size:36px;'>QuantiAR</h1>
<p style='color:gray; font-size:16px;'>
Portfolio Analytics Platform — Backtesting, Real Returns & Macro Benchmarking
</p>
<hr style='border: 1px solid #2c2f36;'>
""", unsafe_allow_html=True)

tab_backtest, tab_markowitz, tab_regresion, tab_news = st.tabs([
    "📊 Backtesting",
    "📈 Markowitz",
    "📉 Regresión",
    "📰 Noticias"
])

END_DATE = datetime.today().strftime("%Y-%m-%d")
MIN_DATA_DATE = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
DEFAULT_ANALYSIS_DATE = "2020-01-01"
WEIGHT_TOLERANCE = 0.001

ETF_NAMES = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF",
    "EEM": "Emerging Markets ETF",
    "GLD": "Gold ETF",
    "SLV": "Silver ETF",
    "VNQ": "Real Estate ETF",
    "XLF": "Financials ETF",
    "XLE": "Energy ETF",
    "XLK": "Technology ETF",
    "XLU": "Utilities ETF",
}

EQUITY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "V": "Visa",
    "MA": "Mastercard",
    "KO": "Coca-Cola",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "NFLX": "Netflix",
    "DIS": "Disney",
    "PFE": "Pfizer",
    "MCD": "McDonald's",
}

BOND_NAMES = {
    "TLT": "Treasury 20+Y",
    "IEF": "Treasury 7-10Y",
    "SHY": "Treasury 1-3Y",
    "LQD": "Corporate IG",
    "HYG": "High Yield Bonds",
    "TIP": "Inflation Protected (TIPS)",
}

ARG_NAMES = {
    "GGAL": "Grupo Galicia",
    "YPF": "YPF",
    "BMA": "Banco Macro",
    "PAM": "Pampa Energía",
    "TGS": "Transportadora Gas del Sur",
    "CEPU": "Central Puerto",
    "EDN": "Edenor",
    "LOMA": "Loma Negra",
    "BBAR": "BBVA Argentina",
    "CRESY": "Cresud",
    "IRS": "IRSA",
    "TEO": "Telecom Argentina",
}

CRYPTO_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
}

NAME_DICT = {}
NAME_DICT.update(ETF_NAMES)
NAME_DICT.update(EQUITY_NAMES)
NAME_DICT.update(BOND_NAMES)
NAME_DICT.update(ARG_NAMES)
NAME_DICT.update(CRYPTO_NAMES)

AVAILABLE_TICKERS = {
    "Merval (ARS)": {
        "YPFD": "YPF",
        "GGAL": "Grupo Galicia",
        "TXAR": "Ternium Argentina",
        "PAMP": "Pampa Energía",
        "ALUA": "Aluar",
        "BBAR": "BBVA Argentina",
        "BMA": "Banco Macro",
        "EDN": "Edenor",
        "TRAN": "Transener",
        "CEPU": "Central Puerto",
        "COME": "Comercial del Plata",
        "CRES": "Cresud",
        "METR": "Metrogas",
        "TGNO4": "Transportadora Gas del Norte",
        "TGSU2": "Transportadora Gas del Sur",
        "SUPV": "Supervielle",
        "BYMA": "BYMA",
        "VALO": "Banco de Valores",
        "LOMA": "Loma Negra",
        "CGPA2": "Ecogas"
    },
    "CEDEARs (ARS)": {
        "AAPL": "Apple",
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "META": "Meta",
        "GOOGL": "Alphabet / Google",
        "AMD": "Advanced Micro Devices",
        "KO": "Coca-Cola",
        "DIS": "Disney",
        "MCD": "McDonald's",
        "BABA": "Alibaba",
        "NFLX": "Netflix",
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "XOM": "Exxon Mobil",
        "INTC": "Intel",
        "PFE": "Pfizer",
        "NKE": "Nike",
        "WMT": "Walmart",
        "GS": "Goldman Sachs",
        "V": "Visa"
    },
    "CEDEAR ETFs (ARS)": {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "GLD": "Gold ETF",
        "SLV": "Silver ETF",
        "XLE": "Energy ETF",
        "EWZ": "Brazil ETF",
        "DIA": "Dow Jones ETF",
        "IWM": "Russell 2000 ETF",
        "TLT": "US Treasury 20Y ETF",
        "EEM": "Emerging Markets ETF"
    }
}

def flatten_available_tickers():
    flat = {}
    for segment, assets in AVAILABLE_TICKERS.items():
        for ticker, name in assets.items():
            flat[ticker] = {
                "name": name,
                "segment": segment,
                "yahoo": f"{ticker}.BA"
            }
    return flat


def format_asset_label(ticker):
    flat = flatten_available_tickers()
    info = flat.get(ticker, {})
    name = info.get("name", ticker)
    segment = info.get("segment", "")
    return f"{ticker} - {name} [{segment}]"


def get_ticker_from_label(label):
    return label.split(" - ")[0]


AVAILABLE_TICKER_MAP = flatten_available_tickers()

NAME_DICT.update({ticker: info["name"] for ticker, info in AVAILABLE_TICKER_MAP.items()})

ASSET_UNIVERSE = {
    "ETFs": {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "DIA": "Dow Jones ETF",
        "IWM": "Russell 2000 ETF",
        "GLD": "Gold ETF",
        "XLF": "Financials ETF",
        "XLE": "Energy ETF",
        "XLK": "Technology ETF",
    },
    "Equity USA / CEDEAR proxy": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "Nvidia",
        "AMZN": "Amazon",
        "GOOGL": "Google",
        "META": "Meta",
        "TSLA": "Tesla",
        "JPM": "JPMorgan",
    },
    "Argentina ADRs": {
        "GGAL": "Grupo Galicia",
        "YPF": "YPF",
        "BMA": "Banco Macro",
        "PAM": "Pampa Energia",
        "CEPU": "Central Puerto",
    },
    "Renta Fija USA": {
        "TLT": "Treasury 20+Y",
        "IEF": "Treasury 7-10Y",
        "LQD": "Corporate IG",
        "HYG": "High Yield",
        "TIP": "Inflation Protected",
    }
}

PORTFOLIOS_FILE = "portfolios.json"


def get_asset_label(ticker):
    if ticker in NAME_DICT:
        return f"{ticker} - {NAME_DICT[ticker]}"

    crypto_names = get_top_crypto()
    if ticker in crypto_names:
        return f"{ticker} - {crypto_names[ticker]}"

    return ticker


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
def get_top_crypto():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 15,
        "page": 1
    }

    data = requests.get(url, params=params, timeout=20).json()

    cryptos = {}
    for coin in data:
        symbol = coin["symbol"].upper()
        name = coin["name"]
        cryptos[symbol] = f"{name}"

    return cryptos


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
    flat_tickers = flatten_available_tickers()

    failed_assets = []

    for asset in selected_assets:
        try:
            yahoo_ticker = flat_tickers[asset]["yahoo"]
            data[asset] = download_yahoo(yahoo_ticker, data_start_date)
        except Exception:
            failed_assets.append(asset)

    prices = pd.DataFrame(data)
    prices = prices.sort_index()
    prices = prices.ffill().dropna(how="all")

    return prices, failed_assets


@st.cache_data
def get_mep_historico():
    url = "https://api.argentinadatos.com/v1/cotizaciones/dolares/bolsa"
    response = requests.get(url, timeout=20)
    data = response.json()

    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha")
    df = df.set_index("fecha")

    df["MEP"] = (df["compra"] + df["venta"]) / 2

    return df["MEP"].dropna()


@st.cache_data
def get_inflacion_arg(start_date):
    url = "https://api.argentinadatos.com/v1/finanzas/indices/inflacion"
    response = requests.get(url, timeout=20)
    data = response.json()

    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha")
    df = df.set_index("fecha")

    inflacion_mensual = df["valor"]
    inflacion_mensual = inflacion_mensual[inflacion_mensual.index >= start_date]

    inflacion_acum = (1 + inflacion_mensual / 100).cumprod()

    return inflacion_acum


def calculate_returns(prices):
    return prices.pct_change().dropna()


def optimize_portfolio(returns, method="sharpe", max_weight=0.5):
    from scipy.optimize import minimize

    n = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_performance(weights):
        ret = np.dot(weights, mean_returns) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return ret, vol

    def neg_sharpe(weights):
        ret, vol = portfolio_performance(weights)
        return -ret / vol if vol != 0 else 0

    def min_vol(weights):
        return portfolio_performance(weights)[1]

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    })

    bounds = tuple((0, max_weight) for _ in range(n))

    init = np.ones(n) / n

    if method == "sharpe":
        result = minimize(neg_sharpe, init, bounds=bounds, constraints=constraints)
    else:
        result = minimize(min_vol, init, bounds=bounds, constraints=constraints)

    return result.x


def simulate_efficient_frontier(returns, num_portfolios=5000):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = returns.shape[1]

    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(n)
        weights = weights / np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol != 0 else np.nan

        results.append({
            "return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe": sharpe,
            "weights": weights
        })

    return pd.DataFrame(results)


def max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return drawdown.min()


def calculate_cagr(series):
    total_return = series.iloc[-1] / series.iloc[0]
    years = (series.index[-1] - series.index[0]).days / 365
    if years <= 0:
        return np.nan
    return total_return ** (1 / years) - 1


def calculate_volatility(series):
    returns = series.pct_change().dropna()
    return returns.std() * np.sqrt(252)


def calculate_sharpe(series):
    returns = series.pct_change().dropna()
    if returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)


def backtest(returns, weights):
    weights = np.array(weights)
    portfolio_returns = returns.dot(weights)
    cumulative = (1 + portfolio_returns).cumprod()
    return portfolio_returns, cumulative


def create_pdf_report(
    selected_assets,
    weights,
    analysis_start,
    analysis_end,
    metrics,
    cartera_ars,
    mep_base,
    inflacion_base,
    corr
):
    buffer = BytesIO()
    navy = "#0B1F33"
    blue = "#00A3E0"
    light_blue = "#D7ECF7"
    gray = "#5F6B7A"
    dark_gray = "#2C2F36"

    with PdfPages(buffer) as pdf:

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.add_patch(plt.Rectangle((0, 0.90), 1, 0.10, color=navy, transform=ax.transAxes))
        ax.text(0.05, 0.945, "QuantiAR", fontsize=28, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(
            0.05, 0.915,
            "Backtesting Report | ARS Portfolio Analytics",
            fontsize=11,
            color=light_blue,
            transform=ax.transAxes
        )

        ax.text(
            0.05, 0.865,
            f"Período analizado: {analysis_start} a {analysis_end}",
            fontsize=11,
            color=gray,
            transform=ax.transAxes
        )

        key_metrics = [
            ("Retorno ARS", metrics["retorno_ars_nominal"]),
            ("Retorno real", metrics["retorno_real"]),
            ("Dólar MEP", metrics["mep_nominal"]),
            ("Inflación", metrics["inflacion"]),
            ("CAGR real", metrics["cagr_real"]),
            ("Sharpe ARS", metrics["sharpe_nominal_ars"]),
            ("Volatilidad ARS", metrics["volatilidad_nominal_ars"]),
            ("Max Drawdown", metrics["max_drawdown_ars"]),
        ]

        x_positions = [0.05, 0.28, 0.51, 0.74]
        y_positions = [0.76, 0.62]

        idx = 0
        for y in y_positions:
            for x in x_positions:
                title, value = key_metrics[idx]

                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        0.20,
                        0.10,
                        facecolor="#F4F7FA",
                        edgecolor="#D8DEE9",
                        linewidth=1,
                        transform=ax.transAxes
                    )
                )

                ax.text(
                    x + 0.01,
                    y + 0.065,
                    title,
                    fontsize=9,
                    color=gray,
                    transform=ax.transAxes
                )

                if "Sharpe" in title:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.2%}"

                ax.text(
                    x + 0.01,
                    y + 0.025,
                    formatted_value,
                    fontsize=15,
                    fontweight="bold",
                    color=blue,
                    transform=ax.transAxes
                )

                idx += 1

        ax.text(
            0.05,
            0.53,
            "Composición del portfolio",
            fontsize=16,
            fontweight="bold",
            color=navy,
            transform=ax.transAxes
        )

        y = 0.49
        for asset, weight in zip(selected_assets, weights):
            label = format_asset_label(asset)

            ax.text(
                0.07,
                y,
                label,
                fontsize=10,
                color=dark_gray,
                transform=ax.transAxes
            )

            ax.text(
                0.78,
                y,
                f"{weight:.2%}",
                fontsize=10,
                fontweight="bold",
                color=navy,
                transform=ax.transAxes
            )

            y -= 0.028

            if y < 0.18:
                ax.text(
                    0.07,
                    y,
                    "...",
                    fontsize=10,
                    color=gray,
                    transform=ax.transAxes
                )
                break

        ax.text(
            0.05,
            0.10,
            "Conclusión",
            fontsize=16,
            fontweight="bold",
            color=navy,
            transform=ax.transAxes
        )

        conclusion_1 = "La cartera generó valor real ajustado por inflación." if metrics["retorno_real"] > 0 else "La cartera no generó valor real ajustado por inflación."
        conclusion_2 = "La cartera superó al dólar MEP." if metrics["retorno_ars_nominal"] > metrics["mep_nominal"] else "La cartera no superó al dólar MEP."

        ax.text(
            0.07,
            0.065,
            f"• {conclusion_1}\n• {conclusion_2}",
            fontsize=11,
            color=dark_gray,
            transform=ax.transAxes
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(cartera_ars, label="Portfolio ARS", color=blue, linewidth=2.8)
        ax.plot(mep_base, label="Dólar MEP", color=navy, linewidth=2.0, linestyle="--")
        ax.plot(inflacion_base, label="Inflación", color="#7A8CA0", linewidth=2.0, linestyle=":")

        ax.set_title(
            "Performance comparada",
            fontsize=18,
            fontweight="bold",
            color=navy,
            pad=18
        )

        ax.set_ylabel("Base 1", fontsize=10, color=gray)
        ax.tick_params(axis="both", labelsize=9, colors=gray)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D8DEE9")
        ax.spines["bottom"].set_color("#D8DEE9")
        ax.legend(frameon=False, fontsize=10)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        fig.patch.set_facecolor("white")

        corr_small = corr.copy()

        im = ax.imshow(
            corr_small.values,
            cmap="Blues",
            vmin=-1,
            vmax=1
        )

        ax.set_title(
            "Matriz de correlación",
            fontsize=18,
            fontweight="bold",
            color=navy,
            pad=18
        )

        ax.set_xticks(np.arange(len(corr_small.columns)))
        ax.set_yticks(np.arange(len(corr_small.index)))
        ax.set_xticklabels(corr_small.columns, rotation=45, ha="right", fontsize=8, color=gray)
        ax.set_yticklabels(corr_small.index, fontsize=8, color=gray)

        for i in range(len(corr_small.index)):
            for j in range(len(corr_small.columns)):
                value = corr_small.iloc[i, j]

                if abs(value) > 0.7:
                    text_color = "white"
                else:
                    text_color = navy

                if abs(value) > 0.8:
                    fontsize = 12
                    weight = "bold"
                elif abs(value) > 0.6:
                    fontsize = 10
                    weight = "bold"
                else:
                    fontsize = 8
                    weight = "normal"

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight=weight,
                    color=text_color
                )

        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors=gray)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer


def create_markowitz_pdf_report(
    selected_assets,
    analysis_start,
    analysis_end,
    max_weight,
    frontier,
    weights_comparison,
    metrics_markowitz
):
    buffer = BytesIO()

    navy = "#0B1F33"
    blue = "#00A3E0"
    gray = "#5F6B7A"
    light_blue = "#D7ECF7"

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.add_patch(plt.Rectangle((0, 0.90), 1, 0.10, color=navy, transform=ax.transAxes))
        ax.text(0.05, 0.945, "QuantiAR", fontsize=28, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(
            0.05, 0.915,
            "Markowitz Optimization Report | Efficient Frontier",
            fontsize=11,
            color=light_blue,
            transform=ax.transAxes
        )

        ax.text(
            0.05,
            0.86,
            f"Período analizado: {analysis_start} a {analysis_end}",
            fontsize=11,
            color=gray,
            transform=ax.transAxes
        )

        ax.text(
            0.05,
            0.825,
            f"Restricción de peso máximo por activo: {max_weight:.0%}",
            fontsize=11,
            color=gray,
            transform=ax.transAxes
        )

        ax.text(
            0.05,
            0.77,
            "Métricas comparadas",
            fontsize=17,
            fontweight="bold",
            color=navy,
            transform=ax.transAxes
        )

        y = 0.72
        for _, row in metrics_markowitz.iterrows():
            ax.add_patch(
                plt.Rectangle(
                    (0.05, y - 0.015),
                    0.90,
                    0.07,
                    facecolor="#F4F7FA",
                    edgecolor="#D8DEE9",
                    linewidth=1,
                    transform=ax.transAxes
                )
            )

            ax.text(0.07, y + 0.025, row["Portfolio"], fontsize=11, fontweight="bold", color=navy, transform=ax.transAxes)
            ax.text(0.32, y + 0.025, f"Retorno: {row['Retorno esperado']:.2%}", fontsize=10, color=gray, transform=ax.transAxes)
            ax.text(0.55, y + 0.025, f"Vol: {row['Volatilidad']:.2%}", fontsize=10, color=gray, transform=ax.transAxes)
            ax.text(0.75, y + 0.025, f"Sharpe: {row['Sharpe']:.2f}", fontsize=10, color=blue, fontweight="bold", transform=ax.transAxes)

            y -= 0.09

        ax.text(
            0.05,
            0.38,
            "Supuestos del modelo",
            fontsize=17,
            fontweight="bold",
            color=navy,
            transform=ax.transAxes
        )

        assumptions = """
• Retornos esperados estimados con promedio histórico diario anualizado.
• Volatilidad y matriz de covarianza anualizadas con 252 ruedas.
• Optimización long-only: no permite posiciones short.
• No considera costos de transacción, impuestos, spreads ni liquidez.
• Los resultados son históricos/simulados y no garantizan retornos futuros.
"""

        ax.text(
            0.07,
            0.34,
            assumptions,
            fontsize=10.5,
            color=gray,
            va="top",
            transform=ax.transAxes
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        scatter = ax.scatter(
            frontier["volatility"],
            frontier["return"],
            c=frontier["sharpe"],
            cmap="Blues",
            alpha=0.65,
            s=12
        )

        ax.set_title("Frontera eficiente", fontsize=18, fontweight="bold", color=navy, pad=18)
        ax.set_xlabel("Volatilidad anualizada", fontsize=10, color=gray)
        ax.set_ylabel("Retorno esperado anualizado", fontsize=10, color=gray)
        ax.tick_params(axis="both", labelsize=9, colors=gray)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Sharpe", color=gray)
        cbar.ax.tick_params(labelsize=8, colors=gray)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(
            0.05,
            0.95,
            "Comparación de pesos",
            fontsize=20,
            fontweight="bold",
            color=navy,
            transform=ax.transAxes
        )

        table_df = weights_comparison.copy()
        table_df["Peso actual"] = table_df["Peso actual"].map(lambda x: f"{x:.2%}")
        table_df["Max Sharpe"] = table_df["Max Sharpe"].map(lambda x: f"{x:.2%}")
        table_df["Min Volatility"] = table_df["Min Volatility"].map(lambda x: f"{x:.2%}")

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.35)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(navy)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#F4F7FA" if row % 2 == 0 else "white")
                cell.set_edgecolor("#D8DEE9")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer


if "selected_assets" not in st.session_state:
    st.session_state.selected_assets = []

if "selected_assets_initialized" not in st.session_state:
    st.session_state.selected_assets_initialized = False


st.sidebar.header("Configuración")

saved_portfolios = load_saved_portfolios()

st.sidebar.subheader("Portfolios guardados")

if saved_portfolios:
    portfolio_to_load = st.sidebar.selectbox(
        "Cargar portfolio",
        ["Nuevo portfolio"] + list(saved_portfolios.keys())
    )
else:
    portfolio_to_load = "Nuevo portfolio"


query_params = st.query_params

url_assets = []
url_weights = []
url_start_date = None

if "assets" in query_params and "weights" in query_params:
    try:
        url_assets = query_params["assets"].split(",")
        url_weights = [float(x) for x in query_params["weights"].split(",")]
    except Exception:
        url_assets = []
        url_weights = []

if "start" in query_params:
    try:
        url_start_date = pd.to_datetime(query_params["start"]).date()
    except Exception:
        url_start_date = None


flat_tickers = flatten_available_tickers()

if url_assets and url_weights:
    default_assets = [a for a in url_assets if a in flat_tickers]
    default_weights = url_weights
    st.session_state.selected_assets = default_assets.copy()

elif portfolio_to_load != "Nuevo portfolio":
    loaded_portfolio = saved_portfolios[portfolio_to_load]
    default_assets = [a for a in loaded_portfolio["assets"] if a in flat_tickers]
    default_weights = loaded_portfolio["weights"]
    st.session_state.selected_assets = default_assets.copy()

else:
    default_assets = []
    default_weights = None


st.sidebar.subheader("Fecha de análisis")

min_date = pd.to_datetime(MIN_DATA_DATE).date()
max_date = pd.to_datetime(END_DATE).date()

if portfolio_to_load != "Nuevo portfolio" and "start_date" in saved_portfolios.get(portfolio_to_load, {}):
    default_date = pd.to_datetime(saved_portfolios[portfolio_to_load]["start_date"]).date()
elif url_start_date:
    default_date = url_start_date
else:
    default_date = pd.to_datetime(DEFAULT_ANALYSIS_DATE).date()

analysis_start_date = st.sidebar.date_input(
    "Analizar desde",
    value=default_date,
    min_value=min_date,
    max_value=max_date
)

st.sidebar.caption("La app trae hasta 10 años de data y analiza desde la fecha elegida.")

st.sidebar.subheader("Universo de activos")

segment = st.sidebar.selectbox(
    "Segmento",
    list(AVAILABLE_TICKERS.keys())
)

segment_assets = AVAILABLE_TICKERS[segment]

asset_options = [
    f"{ticker} - {name}"
    for ticker, name in segment_assets.items()
]

selected_display_assets = st.sidebar.multiselect(
    "Elegí activos del segmento",
    asset_options
)

if "selected_assets" not in st.session_state:
    st.session_state.selected_assets = []

if st.sidebar.button("Agregar activos"):
    for label in selected_display_assets:
        ticker = get_ticker_from_label(label)
        if ticker not in st.session_state.selected_assets:
            st.session_state.selected_assets.append(ticker)

if st.sidebar.button("Reset portfolio"):
    st.session_state.selected_assets = []
    st.session_state.portfolio_weights = {}
    st.session_state.weights_dict = {}
    for key in list(st.session_state.keys()):
        if key.startswith("weight_"):
            del st.session_state[key]
    st.rerun()

selected_assets = st.session_state.selected_assets

st.sidebar.write("### Portfolio actual")
if selected_assets:
    for asset in selected_assets:
        st.sidebar.write(f"- {format_asset_label(asset)}")
else:
    st.sidebar.info("Agregá activos desde un segmento.")

st.sidebar.subheader("Portfolio")

weights = []

if selected_assets:
    if "weights_dict" not in st.session_state:
        st.session_state.weights_dict = {}

    if default_weights:
        for asset, weight in zip(default_assets, default_weights):
            if asset in selected_assets and asset not in st.session_state.weights_dict:
                st.session_state.weights_dict[asset] = weight

    if len(selected_assets) > 0:
        equal_weight = 1 / len(selected_assets)

        for asset in selected_assets:
            if asset not in st.session_state.weights_dict:
                st.session_state.weights_dict[asset] = equal_weight

    for asset in list(st.session_state.weights_dict.keys()):
        if asset not in selected_assets:
            del st.session_state.weights_dict[asset]

    portfolio_df = pd.DataFrame({
        "Asset": selected_assets,
        "Weight (%)": [st.session_state.weights_dict[a] * 100 for a in selected_assets]
    })

    edited_df = st.sidebar.data_editor(
        portfolio_df,
        num_rows="fixed",
        use_container_width=True
    )

    for _, row in edited_df.iterrows():
        asset = row["Asset"]
        weight = row["Weight (%)"] / 100
        st.session_state.weights_dict[asset] = weight

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Equal Weight"):
            for asset in selected_assets:
                st.session_state.weights_dict[asset] = 1 / len(selected_assets)
            st.rerun()

    with col2:
        if st.button("Normalize"):
            total = sum(st.session_state.weights_dict.values())
            if total > 0:
                for asset in selected_assets:
                    st.session_state.weights_dict[asset] /= total
            st.rerun()

    weights = [st.session_state.weights_dict[a] for a in selected_assets]

    weights_sum = sum(weights)
    st.sidebar.write(f"Total: {weights_sum:.2%}")

    portfolio_name = st.sidebar.text_input("Nombre del portfolio")

    col_save, col_run, col_link = st.sidebar.columns(3)

    with col_save:
        save_button = st.button("Guardar")

    with col_run:
        run = st.button("Correr")

    with col_link:
        generate_link = st.button("Link")

    if generate_link and selected_assets:
        assets_str = ",".join(selected_assets)
        weights_str = ",".join([str(round(w, 4)) for w in weights])
        start_str = analysis_start_date.strftime("%Y-%m-%d")

        base_url = "https://quantiar.streamlit.app"
        full_url = f"{base_url}/?assets={assets_str}&weights={weights_str}&start={start_str}"

        st.sidebar.success("Link generado")
        st.sidebar.code(full_url)

    if save_button:
        if not portfolio_name:
            st.sidebar.error("Poné un nombre para guardar.")
        else:
            saved_portfolios[portfolio_name] = {
                "assets": selected_assets,
                "weights": weights,
                "start_date": analysis_start_date.strftime("%Y-%m-%d")
            }

            save_portfolios(saved_portfolios)
            st.sidebar.success(f"Portfolio '{portfolio_name}' guardado.")
            st.rerun()

    with tab_backtest:
        if run:
            with st.spinner("Calculando QuantiAR..."):
                prices, failed_assets = download_prices(selected_assets, MIN_DATA_DATE)

                if failed_assets:
                    st.warning(
                        "No se pudo descargar data para: "
                        + ", ".join(failed_assets)
                        + ". Se excluyeron del cálculo."
                    )

                available_assets = list(prices.columns)

                if len(available_assets) == 0:
                    st.error("No se pudo descargar data para ningún activo seleccionado.")
                    st.stop()

                selected_weights_dict = dict(zip(selected_assets, weights))
                weights = [selected_weights_dict[a] for a in available_assets]

                weights_sum_available = sum(weights)

                if weights_sum_available <= 0:
                    st.error("Los activos con data disponible tienen peso total 0%.")
                    st.stop()

                weights = [w / weights_sum_available for w in weights]
                selected_assets = available_assets

                prices = prices[prices.index >= pd.to_datetime(analysis_start_date)]

                if len(prices) < 30:
                    st.error("No hay suficiente historia de precios para el período seleccionado.")
                    st.stop()

                returns = calculate_returns(prices)

                portfolio_returns, cartera_ars = backtest(returns, weights)

                mep_series = get_mep_historico()
                common_index = cartera_ars.index.intersection(mep_series.index)

                cartera_ars = cartera_ars.loc[common_index]
                mep = mep_series.loc[common_index]
                mep_base = mep / mep.iloc[0]

                inflacion = get_inflacion_arg(cartera_ars.index.min())
                inflacion_diaria = inflacion.reindex(cartera_ars.index, method="ffill").dropna()

                cartera_ars = cartera_ars.loc[inflacion_diaria.index]
                mep_base = mep_base.loc[inflacion_diaria.index]
                inflacion_base = inflacion_diaria / inflacion_diaria.iloc[0]

                ret_ars = cartera_ars.iloc[-1] - 1
                ret_mep = mep_base.iloc[-1] - 1
                ret_inflacion = inflacion_base.iloc[-1] - 1

                ret_real = (cartera_ars.iloc[-1] / inflacion_base.iloc[-1]) - 1
                mep_real = (mep_base.iloc[-1] / inflacion_base.iloc[-1]) - 1

                cagr_nominal_ars = calculate_cagr(cartera_ars)
                cagr_real = calculate_cagr(cartera_ars / inflacion_base)
                vol_nominal_ars = calculate_volatility(cartera_ars)
                sharpe_nominal_ars = calculate_sharpe(cartera_ars)
                mdd_ars = max_drawdown(portfolio_returns.loc[cartera_ars.index])

                st.subheader("Resumen Ejecutivo")

                st.write(
                    f"Período analizado: **{cartera_ars.index.min().date()}** "
                    f"a **{cartera_ars.index.max().date()}**"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Retorno ARS nominal", f"{ret_ars:.2%}")
                col2.metric("Retorno real", f"{ret_real:.2%}")
                col3.metric("CAGR real", f"{cagr_real:.2%}")
                col4.metric("Dólar MEP", f"{ret_mep:.2%}")

                col5, col6, col7, col8 = st.columns(4)

                col5.metric("Inflación", f"{ret_inflacion:.2%}")
                col6.metric("Sharpe ARS", f"{sharpe_nominal_ars:.2f}")
                col7.metric("Volatilidad ARS", f"{vol_nominal_ars:.2%}")
                col8.metric("Max Drawdown ARS", f"{mdd_ars:.2%}")

                if ret_real > 0:
                    st.success("✅ La cartera genera valor real ajustado por inflación.")
                else:
                    st.error("❌ La cartera no genera valor real ajustado por inflación.")

                if ret_ars > ret_mep:
                    st.success("✅ La cartera le gana al dólar MEP.")
                else:
                    st.warning("⚠️ La cartera no le gana al dólar MEP.")

                st.subheader("Performance comparada")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=cartera_ars.index,
                    y=cartera_ars,
                    mode="lines",
                    name="Portfolio ARS",
                    line=dict(width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=mep_base.index,
                    y=mep_base,
                    mode="lines",
                    name="Dólar MEP",
                    line=dict(width=2, dash="dash")
                ))

                fig.add_trace(go.Scatter(
                    x=inflacion_base.index,
                    y=inflacion_base,
                    mode="lines",
                    name="Inflación",
                    line=dict(width=2, dash="dot")
                ))

                fig.update_layout(
                    title="Portfolio ARS vs Dólar MEP vs Inflación",
                    template="plotly_dark",
                    height=520,
                    hovermode="x unified",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(color="#FAFAFA"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=20, r=20, t=70, b=20)
                )

                fig.update_yaxes(title="Base 1", gridcolor="#2c2f36")
                fig.update_xaxes(gridcolor="#2c2f36")

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Matriz de correlación")

                corr = returns.corr()
                st.dataframe(corr)

                st.subheader("Data exportable")

                results_df = pd.DataFrame([{
                    "fecha_inicio": cartera_ars.index.min().date(),
                    "fecha_fin": cartera_ars.index.max().date(),
                    "retorno_ars_nominal": ret_ars,
                    "retorno_real": ret_real,
                    "mep_nominal": ret_mep,
                    "mep_real": mep_real,
                    "inflacion": ret_inflacion,
                    "cagr_nominal_ars": cagr_nominal_ars,
                    "cagr_real": cagr_real,
                    "volatilidad_nominal_ars": vol_nominal_ars,
                    "sharpe_nominal_ars": sharpe_nominal_ars,
                    "max_drawdown_ars": mdd_ars
                }])

                st.dataframe(results_df)

                metrics_for_pdf = results_df.iloc[0].to_dict()

                pdf_buffer = create_pdf_report(
                    selected_assets=selected_assets,
                    weights=weights,
                    analysis_start=cartera_ars.index.min().date(),
                    analysis_end=cartera_ars.index.max().date(),
                    metrics=metrics_for_pdf,
                    cartera_ars=cartera_ars,
                    mep_base=mep_base,
                    inflacion_base=inflacion_base,
                    corr=corr
                )

                st.download_button(
                    label="📄 Exportar PDF",
                    data=pdf_buffer,
                    file_name="quantiar_backtesting_ars_report.pdf",
                    mime="application/pdf"
                )

else:
    with tab_backtest:
        st.info("Elegí al menos un activo para empezar.")

with tab_markowitz:
    st.subheader("Optimización de Portfolio - Markowitz")
    st.caption(
        "Modelo long-only basado en retornos históricos, volatilidad y correlación. "
        "No contempla costos, impuestos, spreads ni liquidez."
    )

    max_weight_pct = st.slider(
        "Peso máximo por activo",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        format="%d%%"
    )

    max_weight = max_weight_pct / 100

    if len(selected_assets) < 2:
        st.warning("Seleccioná al menos 2 activos en Backtesting.")
    else:
        prices, failed_assets = download_prices(selected_assets, MIN_DATA_DATE)
        prices = prices[prices.index >= pd.to_datetime(analysis_start_date)]

        if failed_assets:
            st.warning("Algunos activos no pudieron descargarse: " + ", ".join(failed_assets))

        prices = prices.dropna(axis=1, how="all")
        returns = calculate_returns(prices)

        available_assets = list(returns.columns)

        if len(available_assets) < 2:
            st.error("No hay suficientes activos con data disponible para optimizar.")
            st.stop()

        current_weights_dict = st.session_state.get("weights_dict", {})
        current_weights = np.array([
            current_weights_dict.get(asset, 1 / len(available_assets))
            for asset in available_assets
        ])

        current_weights = current_weights / current_weights.sum()

        st.write(
            f"Período utilizado: **{prices.index.min().date()}** "
            f"a **{prices.index.max().date()}**"
        )

        frontier = simulate_efficient_frontier(returns, num_portfolios=5000)

        max_sharpe_weights = optimize_portfolio(returns, "sharpe", max_weight=max_weight)
        min_vol_weights = optimize_portfolio(returns, "vol", max_weight=max_weight)

        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        def portfolio_stats(weights):
            ret = np.dot(weights, mean_returns) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = ret / vol if vol != 0 else np.nan
            return ret, vol, sharpe

        current_ret, current_vol, current_sharpe = portfolio_stats(current_weights)
        max_ret, max_vol, max_sharpe = portfolio_stats(max_sharpe_weights)
        min_ret, min_vol, min_sharpe = portfolio_stats(min_vol_weights)

        st.subheader("Frontera eficiente")

        fig_frontier = go.Figure()

        fig_frontier.add_trace(go.Scatter(
            x=frontier["volatility"],
            y=frontier["return"],
            mode="markers",
            name="Carteras simuladas",
            marker=dict(
                size=7,
                color=frontier["sharpe"],
                colorscale=[
                    [0.0, "#1C1F26"],
                    [0.35, "#1F6F8B"],
                    [0.70, "#00A3E0"],
                    [1.0, "#D7ECF7"]
                ],
                showscale=True,
                colorbar=dict(
                    title="Sharpe",
                    thickness=14,
                    len=0.75
                ),
                opacity=0.72,
                line=dict(width=0)
            ),
            text=[
                f"<b>Cartera simulada</b><br>"
                f"Sharpe: {s:.2f}<br>"
                f"Retorno esperado: {r:.2%}<br>"
                f"Volatilidad: {v:.2%}"
                for s, r, v in zip(frontier["sharpe"], frontier["return"], frontier["volatility"])
            ],
            hoverinfo="text"
        ))

        fig_frontier.add_trace(go.Scatter(
            x=[max_vol],
            y=[max_ret],
            mode="markers+text",
            name="Max Sharpe",
            marker=dict(
                size=22,
                symbol="star",
                color="#00A3E0",
                line=dict(color="white", width=1.5)
            ),
            text=["Max Sharpe"],
            textposition="top center",
            hovertext=[
                f"<b>Max Sharpe</b><br>"
                f"Retorno esperado: {max_ret:.2%}<br>"
                f"Volatilidad: {max_vol:.2%}<br>"
                f"Sharpe: {max_sharpe:.2f}"
            ],
            hoverinfo="text"
        ))

        fig_frontier.add_trace(go.Scatter(
            x=[min_vol],
            y=[min_ret],
            mode="markers+text",
            name="Min Volatility",
            marker=dict(
                size=18,
                symbol="diamond",
                color="#D7ECF7",
                line=dict(color="#0B1F33", width=1.5)
            ),
            text=["Min Vol"],
            textposition="bottom center",
            hovertext=[
                f"<b>Min Volatility</b><br>"
                f"Retorno esperado: {min_ret:.2%}<br>"
                f"Volatilidad: {min_vol:.2%}<br>"
                f"Sharpe: {min_sharpe:.2f}"
            ],
            hoverinfo="text"
        ))

        fig_frontier.add_trace(go.Scatter(
            x=[current_vol],
            y=[current_ret],
            mode="markers+text",
            name="Portfolio actual",
            marker=dict(
                size=18,
                symbol="circle",
                color="#F39C12",
                line=dict(color="white", width=1.5)
            ),
            text=["Actual"],
            textposition="top right",
            hovertext=[
                f"<b>Portfolio actual</b><br>"
                f"Retorno esperado: {current_ret:.2%}<br>"
                f"Volatilidad: {current_vol:.2%}<br>"
                f"Sharpe: {current_sharpe:.2f}"
            ],
            hoverinfo="text"
        ))

        fig_frontier.update_layout(
            title=dict(
                text="Frontera eficiente — Retorno esperado vs Volatilidad",
                font=dict(size=22, color="#FAFAFA")
            ),
            template="plotly_dark",
            height=650,
            hovermode="closest",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=30, r=30, t=90, b=40)
        )

        fig_frontier.update_xaxes(
            title="Volatilidad anualizada",
            tickformat=".0%",
            gridcolor="#2c2f36",
            zeroline=False
        )

        fig_frontier.update_yaxes(
            title="Retorno esperado anualizado",
            tickformat=".0%",
            gridcolor="#2c2f36",
            zeroline=False
        )

        st.plotly_chart(fig_frontier, use_container_width=True)

        st.subheader("Comparación de pesos")

        weights_comparison = pd.DataFrame({
            "Activo": available_assets,
            "Peso actual": current_weights,
            "Max Sharpe": max_sharpe_weights,
            "Min Volatility": min_vol_weights
        })

        weights_display = weights_comparison.copy()
        weights_display["Peso actual"] = weights_display["Peso actual"].map(lambda x: f"{x:.2%}")
        weights_display["Max Sharpe"] = weights_display["Max Sharpe"].map(lambda x: f"{x:.2%}")
        weights_display["Min Volatility"] = weights_display["Min Volatility"].map(lambda x: f"{x:.2%}")

        st.dataframe(weights_display)

        col_apply_1, col_apply_2 = st.columns(2)

        with col_apply_1:
            if st.button("Aplicar Max Sharpe"):
                for asset, weight in zip(available_assets, max_sharpe_weights):
                    st.session_state.weights_dict[asset] = weight
                    st.session_state[f"weight_{asset}"] = float(weight * 100)
                st.success("Pesos Max Sharpe aplicados.")
                st.rerun()

        with col_apply_2:
            if st.button("Aplicar Min Volatility"):
                for asset, weight in zip(available_assets, min_vol_weights):
                    st.session_state.weights_dict[asset] = weight
                    st.session_state[f"weight_{asset}"] = float(weight * 100)
                st.success("Pesos Min Volatility aplicados.")
                st.rerun()

        st.subheader("Métricas comparadas")

        metrics_markowitz = pd.DataFrame([
            {
                "Portfolio": "Actual",
                "Retorno esperado": current_ret,
                "Volatilidad": current_vol,
                "Sharpe": current_sharpe
            },
            {
                "Portfolio": "Max Sharpe",
                "Retorno esperado": max_ret,
                "Volatilidad": max_vol,
                "Sharpe": max_sharpe
            },
            {
                "Portfolio": "Min Volatility",
                "Retorno esperado": min_ret,
                "Volatilidad": min_vol,
                "Sharpe": min_sharpe
            }
        ])

        metrics_display = metrics_markowitz.copy()
        metrics_display["Retorno esperado"] = metrics_display["Retorno esperado"].map(lambda x: f"{x:.2%}")
        metrics_display["Volatilidad"] = metrics_display["Volatilidad"].map(lambda x: f"{x:.2%}")
        metrics_display["Sharpe"] = metrics_display["Sharpe"].map(lambda x: f"{x:.2f}")

        st.dataframe(metrics_display)

        pdf_markowitz = create_markowitz_pdf_report(
            selected_assets=available_assets,
            analysis_start=prices.index.min().date(),
            analysis_end=prices.index.max().date(),
            max_weight=max_weight,
            frontier=frontier,
            weights_comparison=weights_comparison,
            metrics_markowitz=metrics_markowitz
        )

        st.download_button(
            label="📄 Exportar PDF Markowitz",
            data=pdf_markowitz,
            file_name="quantiar_markowitz_report.pdf",
            mime="application/pdf"
        )

with tab_regresion:
    st.subheader("Regresión Lineal")
    st.info("Próximamente")

with tab_news:
    st.subheader("Noticias de Mercado")
    st.info("Próximamente")
