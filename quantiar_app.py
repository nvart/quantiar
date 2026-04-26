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


def weights_are_valid(weights_sum):
    return abs(weights_sum - 1.0) <= WEIGHT_TOLERANCE


def create_pdf_report(
    selected_assets,
    weights,
    analysis_start,
    analysis_end,
    metrics,
    cartera_ars,
    mep_norm,
    inflacion_base,
    cumulative_usd,
    corr
):
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("QuantiAR - Portfolio Report", fontsize=22, fontweight="bold", y=0.96)

        ax = fig.add_subplot(111)
        ax.axis("off")

        portfolio_text = "\n".join([
            f"{asset}: {weight:.2%}"
            for asset, weight in zip(selected_assets, weights)
        ])

        summary_text = f"""
Período analizado: {analysis_start} a {analysis_end}

COMPOSICIÓN DEL PORTFOLIO
{portfolio_text}

PERFORMANCE
Retorno USD: {metrics['retorno_usd']:.2%}
Retorno ARS nominal: {metrics['retorno_ars_nominal']:.2%}
Retorno real: {metrics['retorno_real']:.2%}
CAGR real: {metrics['cagr_real']:.2%}

BENCHMARKS
Dólar MEP: {metrics['mep_nominal']:.2%}
Inflación: {metrics['inflacion']:.2%}
MEP real: {metrics['mep_real']:.2%}

RIESGO
Sharpe nominal ARS: {metrics['sharpe_nominal_ars']:.2f}
Volatilidad nominal ARS: {metrics['volatilidad_nominal_ars']:.2%}
Max Drawdown USD: {metrics['max_drawdown_usd']:.2%}
"""

        ax.text(
            0.05,
            0.90,
            summary_text,
            fontsize=12,
            va="top",
            family="monospace"
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cartera_ars, label="Cartera ARS")
        ax.plot(mep_norm, label="Dólar MEP")
        ax.plot(inflacion_base, label="Inflación")
        ax.set_title("Cartera vs Dólar MEP vs Inflación")
        ax.set_ylabel("Base 1")
        ax.grid(True)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative_usd, label="Cartera USD")
        ax.set_title("Performance de la cartera en USD")
        ax.set_ylabel("Base 1")
        ax.grid(True)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.index)

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

        ax.set_title("Matriz de correlación")
        fig.colorbar(im, ax=ax)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer


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


if url_assets and url_weights:
    default_assets = [a for a in url_assets if a in AVAILABLE_TICKERS]
    default_weights = url_weights

elif portfolio_to_load != "Nuevo portfolio":
    loaded_portfolio = saved_portfolios[portfolio_to_load]
    default_assets = loaded_portfolio["assets"]
    default_weights = loaded_portfolio["weights"]

else:
    default_assets = ["SPY", "QQQ", "GLD", "BTC"]
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


selected_assets = st.sidebar.multiselect(
    "Elegí activos",
    list(AVAILABLE_TICKERS.keys()),
    default=default_assets
)

st.sidebar.subheader("Pesos del portfolio")

weights = []

if selected_assets:

    if "portfolio_weights" not in st.session_state:
        st.session_state.portfolio_weights = {}

    if default_weights:
        for asset, weight in zip(default_assets, default_weights):
            if asset in selected_assets:
                st.session_state.portfolio_weights[asset] = weight
                st.session_state[f"weight_{asset}"] = float(weight * 100)

    if st.sidebar.button("Equal Weight"):
        equal_pct = round(100 / len(selected_assets), 4)

        for asset in selected_assets:
            st.session_state[f"weight_{asset}"] = equal_pct
            st.session_state.portfolio_weights[asset] = equal_pct / 100

        st.rerun()

    for asset in selected_assets:
        default_value = st.session_state.portfolio_weights.get(
            asset,
            1 / len(selected_assets)
        )

        input_key = f"weight_{asset}"

        if input_key not in st.session_state:
            st.session_state[input_key] = float(default_value * 100)

        weight_pct = st.sidebar.number_input(
            f"{asset} (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key=input_key
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
        elif not weights_are_valid(weights_sum):
            st.sidebar.error("Los pesos deben sumar 100%.")
        else:
            saved_portfolios[portfolio_name] = {
                "assets": selected_assets,
                "weights": weights,
                "start_date": analysis_start_date.strftime("%Y-%m-%d")
            }

            save_portfolios(saved_portfolios)
            st.sidebar.success(f"Portfolio '{portfolio_name}' guardado.")
            st.rerun()

    if not weights_are_valid(weights_sum):
        st.sidebar.error("Los pesos deben sumar 100%.")
    else:
        if run:
            with st.spinner("Calculando QuantiAR..."):

                prices = download_prices(selected_assets, MIN_DATA_DATE)
                prices = prices[prices.index >= pd.to_datetime(analysis_start_date)]

                returns = calculate_returns(prices)

                portfolio_returns, cumulative_usd = backtest(returns, weights)

                mep_series = get_mep_historico()

                common_index = cumulative_usd.index.intersection(mep_series.index)

                cumulative_usd = cumulative_usd.loc[common_index]
                mep = mep_series.loc[common_index]
                mep_norm = mep / mep.iloc[0]

                cartera_ars = cumulative_usd * mep_norm

                inflacion = get_inflacion_arg(cartera_ars.index.min())
                inflacion_diaria = inflacion.reindex(cartera_ars.index, method="ffill").dropna()

                cartera_ars = cartera_ars.loc[inflacion_diaria.index]
                mep_norm = mep_norm.loc[inflacion_diaria.index]
                inflacion_base = inflacion_diaria / inflacion_diaria.iloc[0]

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

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=cartera_ars.index,
                    y=cartera_ars,
                    mode="lines",
                    name="Portfolio ARS",
                    line=dict(width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=mep_norm.index,
                    y=mep_norm,
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
                    title="Portfolio vs Dólar MEP vs Inflación",
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

                st.subheader("Performance en USD")

                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(
                    x=cumulative_usd.index,
                    y=cumulative_usd,
                    mode="lines",
                    name="Portfolio USD",
                    line=dict(width=3)
                ))

                fig2.update_layout(
                    title="Portfolio Performance in USD",
                    template="plotly_dark",
                    height=480,
                    hovermode="x unified",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(color="#FAFAFA"),
                    margin=dict(l=20, r=20, t=60, b=20)
                )

                fig2.update_yaxes(title="Base 1", gridcolor="#2c2f36")
                fig2.update_xaxes(gridcolor="#2c2f36")

                st.plotly_chart(fig2, use_container_width=True)

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

                metrics_for_pdf = results_df.iloc[0].to_dict()

                pdf_buffer = create_pdf_report(
                    selected_assets=selected_assets,
                    weights=weights,
                    analysis_start=cartera_ars.index.min().date(),
                    analysis_end=cartera_ars.index.max().date(),
                    metrics=metrics_for_pdf,
                    cartera_ars=cartera_ars,
                    mep_norm=mep_norm,
                    inflacion_base=inflacion_base,
                    cumulative_usd=cumulative_usd,
                    corr=corr
                )

                st.download_button(
                    label="📄 Exportar PDF",
                    data=pdf_buffer,
                    file_name="quantiar_portfolio_report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("Elegí al menos un activo para empezar.")
