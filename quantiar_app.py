import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="QuantiAR", layout="wide")

st.title("📊 QuantiAR - Portfolio Analytics")
st.caption("Backtesting de carteras con comparación vs Dólar MEP e inflación argentina.")

END_DATE = datetime.today().strftime("%Y-%m-%d")

# Traemos hasta 10 años de data
MIN_DATA_DATE = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
DEFAULT_ANALYSIS_DATE = "2020-01-01"


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

    with open(PORTFOLIOS_FILE, "r") as file:
        return json.load(file)


def save_portfolios(portfolios):
    with open(PORTFOLIOS_FILE, "w") as file:
        json.dump(portfolios, file, indent=4)


# =========================
# DATA
# =========================

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


# =========================
# FINANCE FUNCTIONS
# =========================

def calculate_returns(prices):
    return prices.pct_change().dropna()


def max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return drawdown.min()


def calculate_cagr(series):
    total_return = series.iloc[-1] / series.iloc[0]
    years = (series.index[-1] - series.index[0]).days / 365
    return total_return ** (1 / years) - 1


def calculate_volatility(series):
    returns = series.pct_change().dropna()
    return returns.std() * np.sqrt(252)


def calculate_sharpe(series):
    returns = series.pct_change().dropna()
    return returns.mean() / returns.std() * np.sqrt(252)


def backtest(returns, weights):
    weights = np.array(weights)
    portfolio_returns = returns.dot(weights)
    cumulative = (1 + portfolio_returns).cumprod()
    return portfolio_returns, cumulative

# =========================
# UI
# =========================

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


# =========================
# LEER PARAMETROS DE URL
# =========================

query_params = st.query_params

url_assets = []
url_weights = []
url_start_date = None

if "assets" in query_params and "weights" in query_params:
    try:
        url_assets = query_params["assets"].split(",")
        url_weights = [float(x) for x in query_params["weights"].split(",")]
    except:
        url_assets = []
        url_weights = []

if "start" in query_params:
    try:
        url_start_date = pd.to_datetime(query_params["start"]).date()
    except:
        url_start_date = None


if url_assets and url_weights:
    default_assets = url_assets
    default_weights = url_weights

elif portfolio_to_load != "Nuevo portfolio":
    loaded_portfolio = saved_portfolios[portfolio_to_load]
    default_assets = loaded_portfolio["assets"]
    default_weights = loaded_portfolio["weights"]

else:
    default_assets = ["SPY", "QQQ", "GLD", "BTC"]
    default_weights = None


# =========================
# FECHA DE ANÁLISIS
# =========================

st.sidebar.subheader("Fecha de análisis")

min_date = pd.to_datetime(MIN_DATA_DATE).date()
max_date = pd.to_datetime(END_DATE).date()

default_date = url_start_date if url_start_date else pd.to_datetime(DEFAULT_ANALYSIS_DATE).date()

analysis_start_date = st.sidebar.date_input(
    "Analizar desde",
    value=default_date,
    min_value=min_date,
    max_value=max_date
)

st.sidebar.caption("La app trae hasta 10 años de data y analiza desde la fecha elegida.")


# =========================
# ACTIVOS
# =========================

selected_assets = st.sidebar.multiselect(
    "Elegí activos",
    list(AVAILABLE_TICKERS.keys()),
    default=default_assets
)

st.sidebar.subheader("Pesos del portfolio")

weights = []

if selected_assets:

    # Inicializamos pesos en session_state
    if "portfolio_weights" not in st.session_state:
        st.session_state.portfolio_weights = {}

    # Cargar pesos desde portfolio guardado o URL
    if default_weights:
        for asset, weight in zip(default_assets, default_weights):
            if asset in selected_assets:
                st.session_state.portfolio_weights[asset] = weight

    # Botón Equal Weight
    if st.sidebar.button("Equal Weight"):
        equal_weight = 1 / len(selected_assets)
        for asset in selected_assets:
            st.session_state.portfolio_weights[asset] = equal_weight

    for asset in selected_assets:
        default_value = st.session_state.portfolio_weights.get(
            asset,
            1 / len(selected_assets)
        )

        weight_pct = st.sidebar.number_input(
            f"{asset} (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_value * 100),
            step=5.0,
            key=f"weight_{asset}"
        )

        weight = weight_pct / 100
        st.session_state.portfolio_weights[asset] = weight
        weights.append(weight)

    weights_sum = sum(weights)

    st.sidebar.write(f"**Suma de pesos:** {weights_sum:.2%}")

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
        elif not np.isclose(weights_sum, 1.0):
            st.sidebar.error("Los pesos deben sumar 100%.")
        else:
            saved_portfolios[portfolio_name] = {
                "assets": selected_assets,
                "weights": weights,
                "start_date": analysis_start_date.strftime("%Y-%m-%d")
            }

            save_portfolios(saved_portfolios)
            st.sidebar.success(f"Portfolio '{portfolio_name}' guardado.")


    if not np.isclose(weights_sum, 1.0):
        st.sidebar.error("Los pesos deben sumar 100%.")
    else:
        if run:
            with st.spinner("Calculando QuantiAR..."):

                prices = download_prices(selected_assets, MIN_DATA_DATE)

                # Filtramos desde fecha elegida por el usuario
                prices = prices[prices.index >= pd.to_datetime(analysis_start_date)]

                returns = calculate_returns(prices)

                portfolio_returns, cumulative_usd = backtest(returns, weights)

                # MEP histórico
                mep_series = get_mep_historico()

                common_index = cumulative_usd.index.intersection(mep_series.index)

                cumulative_usd = cumulative_usd.loc[common_index]
                mep = mep_series.loc[common_index]
                mep_norm = mep / mep.iloc[0]

                cartera_ars = cumulative_usd * mep_norm

                # Inflación
                inflacion = get_inflacion_arg(cartera_ars.index.min())
                inflacion_diaria = inflacion.reindex(cartera_ars.index, method="ffill").dropna()

                cartera_ars = cartera_ars.loc[inflacion_diaria.index]
                mep_norm = mep_norm.loc[inflacion_diaria.index]
                inflacion_base = inflacion_diaria / inflacion_diaria.iloc[0]

                # Métricas
                ret_usd = cumulative_usd.loc[cartera_ars.index].iloc[-1] - 1
                ret_ars = cartera_ars.iloc[-1] - 1
                ret_mep = mep_norm.iloc[-1] - 1
                ret_inflacion = inflacion_base.iloc[-1] - 1

                ret_real = (cartera_ars.iloc[-1] / inflacion_base.iloc[-1]) - 1
                mep_real = (mep_norm.iloc[-1] / inflacion_base.iloc[-1]) - 1

                cagr_nominal_ars = calculate_cagr(cartera_ars)
                cagr_real = calculate_cagr(cartera_ars / inflacion_base)
                vol_nominal_ars = calculate_volatility(cartera_ars)
                sharpe_nominal_ars = calculate_sharpe(cartera_ars)
                mdd_usd = max_drawdown(portfolio_returns.loc[cumulative_usd.index])

                st.subheader("Resumen Ejecutivo")

                st.write(
                    f"Período analizado: **{cartera_ars.index.min().date()}** "
                    f"a **{cartera_ars.index.max().date()}**"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Retorno USD", f"{ret_usd:.2%}")
                col2.metric("Retorno ARS nominal", f"{ret_ars:.2%}")
                col3.metric("Retorno real", f"{ret_real:.2%}")
                col4.metric("CAGR real", f"{cagr_real:.2%}")

                col5, col6, col7, col8 = st.columns(4)

                col5.metric("Dólar MEP", f"{ret_mep:.2%}")
                col6.metric("Inflación", f"{ret_inflacion:.2%}")
                col7.metric("Sharpe nominal ARS", f"{sharpe_nominal_ars:.2f}")
                col8.metric("Max Drawdown USD", f"{mdd_usd:.2%}")

                if ret_real > 0:
                    st.success("✅ La cartera genera valor real.")
                else:
                    st.error("❌ La cartera no genera valor real.")

                if ret_ars > ret_mep:
                    st.success("✅ La cartera le gana al dólar MEP.")
                else:
                    st.warning("⚠️ La cartera no le gana al dólar MEP.")

                st.subheader("Performance comparada")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(cartera_ars, label="Cartera ARS")
                ax.plot(mep_norm, label="Dólar MEP")
                ax.plot(inflacion_base, label="Inflación")
                ax.set_title("Cartera vs Dólar MEP vs Inflación")
                ax.set_ylabel("Base 1")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                st.subheader("Performance en USD")

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(cumulative_usd, label="Cartera USD")
                ax2.set_title("Cartera en USD")
                ax2.set_ylabel("Base 1")
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)
        
                st.subheader("Matriz de correlación")

                corr = returns.corr()
                st.dataframe(corr)

                st.subheader("Data exportable")

                results_df = pd.DataFrame([{
                    "fecha_inicio": cartera_ars.index.min().date(),
                    "fecha_fin": cartera_ars.index.max().date(),
                    "retorno_usd": ret_usd,
                    "retorno_ars_nominal": ret_ars,
                    "retorno_real": ret_real,
                    "mep_nominal": ret_mep,
                    "mep_real": mep_real,
                    "inflacion": ret_inflacion,
                    "cagr_nominal_ars": cagr_nominal_ars,
                    "cagr_real": cagr_real,
                    "volatilidad_nominal_ars": vol_nominal_ars,
                    "sharpe_nominal_ars": sharpe_nominal_ars,
                    "max_drawdown_usd": mdd_usd
                }])

                st.dataframe(results_df)

else:
    st.info("Elegí al menos un activo para empezar.")

