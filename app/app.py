# app/app.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.express as px  # novos gráficos

DATA_DIR = Path("data/processed/")
CSV_PATH = DATA_DIR / "daily_weather.csv"
PARQUET_PATH = DATA_DIR / "daily_weather.parquet"

st.set_page_config(page_title="Weather ETL • Dashboard", page_icon="🌤️", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Carrega dados diários; tenta CSV e depois Parquet."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    elif PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    else:
        raise FileNotFoundError(
            "Nenhum arquivo encontrado em data/processed/. "
            "Rode o pipeline primeiro: `python -m src.etl.run_all`"
        )

    # Tipos e ordenação
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "city" in df.columns:
        df["city"] = df["city"].astype("string")

    cols = ["city", "date", "temp_min", "temp_avg", "temp_max", "rh_avg", "precip_sum", "hours"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["city", "date"])
    return df


def kpi(value: Optional[float], suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "—"


st.sidebar.header("Filtros")

try:
    df = load_data()

    # ---- Filtros ----
    cities = sorted(df["city"].dropna().unique().tolist())
    sel_cities = st.sidebar.multiselect("Cidades", options=cities, default=cities[:3] if cities else [])
    min_date, max_date = df["date"].min(), df["date"].max()
    sel_dates = st.sidebar.date_input(
        "Período",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(sel_dates, tuple):
        start_date, end_date = sel_dates
    else:
        start_date = sel_dates
        end_date = sel_dates

    # Aplica filtro
    mask = (
        df["city"].isin(sel_cities) if sel_cities else pd.Series([True] * len(df))
    ) & (df["date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))
    dff = df.loc[mask].copy()

    st.title("🌤️ Weather ETL • Daily Dashboard")

    if dff.empty:
        st.info("Nenhum dado para os filtros selecionados. Ajuste as cidades e o período.")
        st.stop()

    # ---- KPIs ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperatura média (°C)", kpi(dff["temp_avg"].mean()))
    with col2:
        st.metric("Temperatura máx (°C)", kpi(dff["temp_max"].max()))
    with col3:
        st.metric("Umidade média (%)", kpi(dff["rh_avg"].mean()))
    with col4:
        st.metric("Precipitação total (mm)", kpi(dff["precip_sum"].sum()))

    # ---- Visualizações ----
    st.subheader("Visualizações")
    tab1, tab2,  tab4, tab5 = st.tabs(
        [
            "Temperatura (linha)",
            "Umidade (linha)",
            # "Precipitação (área)",
            "Distribuição (boxplot)",
            "Correlação (dispersão)",
        ]
    )

    # Temperatura (linha)
    with tab1:
        st.caption("Médias diárias por cidade")
        temp_pivot = (
            dff.pivot_table(index="date", columns="city", values="temp_avg", aggfunc="mean").sort_index()
        )
        st.line_chart(temp_pivot, height=300)

    # Umidade (linha)
    with tab2:
        st.caption("Médias diárias por cidade")
        rh_pivot = dff.pivot_table(index="date", columns="city", values="rh_avg", aggfunc="mean").sort_index()
        st.line_chart(rh_pivot, height=300)

    # Precipitação (área)
    # with tab3:
    #     st.caption("Acumulado diário por cidade")
    #     pr_pivot = dff.pivot_table(index="date", columns="city", values="precip_sum", aggfunc="sum").sort_index()
    #     st.area_chart(pr_pivot, height=300)

    # Distribuição (boxplot)
    with tab4:
        st.caption("Distribuição diária por cidade")
        metric = st.selectbox(
            "Métrica para boxplot",
            options=["temp_avg", "temp_min", "temp_max", "rh_avg", "precip_sum"],
            index=0,
            key="box_metric",
        )
        dbox = dff[["city", metric]].dropna()
        fig_box = px.box(
            dbox,
            x="city",
            y=metric,
            points="outliers",
            title=f"Boxplot de {metric} por cidade",
            labels={"city": "Cidade", metric: metric},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Correlação (dispersão)
    with tab5:
        st.caption("Correlação entre variáveis")
        x_var = st.selectbox(
            "Eixo X",
            options=["temp_avg", "temp_min", "temp_max", "rh_avg", "precip_sum"],
            index=0,
            key="scatter_x",
        )
        y_var = st.selectbox(
            "Eixo Y",
            options=["temp_avg", "temp_min", "temp_max", "rh_avg", "precip_sum"],
            index=3,
            key="scatter_y",
        )
        color_by = st.selectbox(
            "Colorir por",
            options=["city", None],
            index=0,
            key="scatter_color",
        )

        dplot = dff[[x_var, y_var, "city"]].dropna()
        fig_scatter = px.scatter(
            dplot,
            x=x_var,
            y=y_var,
            color="city" if color_by == "city" else None,
            trendline="ols",  # remova se não instalar statsmodels
            title=f"Dispersão: {x_var} vs {y_var}",
            labels={x_var: x_var, y_var: y_var},
            opacity=0.7,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Tabela ----
    st.subheader("Tabela filtrada")
    st.dataframe(dff.reset_index(drop=True), use_container_width=True)

    # ---- Download ----
    st.subheader("Exportar dados filtrados")
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV", data=csv_bytes, file_name="weather_filtered.csv", mime="text/csv")

except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# from pathlib import Path
# import pandas as pd
# import streamlit as st

# DATA_DIR = Path("data")
# CSV_PATH = DATA_DIR / "daily_weather.csv"
# PARQUET_PATH = DATA_DIR / "daily_weather.parquet"

# st.set_page_config(page_title="Weather Dashboard", page_icon="🌤️", layout="wide")

# @st.cache_data(show_spinner=False)
# def load_data() -> pd.DataFrame:
#     if CSV_PATH.exists():
#         df = pd.read_csv(CSV_PATH)
#     elif PARQUET_PATH.exists():
#         df = pd.read_parquet(PARQUET_PATH)
#     else:
#         raise FileNotFoundError("Nenhum arquivo encontrado em data/processed/. Rode o ETL antes.")
#     if "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"])
#     return df

# st.sidebar.header("Filtros")

# try:
#     df = load_data()
#     cities = sorted(df["city"].dropna().unique().tolist())
#     sel_cities = st.sidebar.multiselect("Cidades", options=cities, default=cities)
#     min_date, max_date = df["date"].min(), df["date"].max()
#     sel_dates = st.sidebar.date_input("Período", value=(min_date, max_date))

#     if isinstance(sel_dates, tuple):
#         start_date, end_date = sel_dates
#     else:
#         start_date = sel_dates
#         end_date = sel_dates

#     # mask = df["city"].isin(sel_cities) & df["date"].between(start_date, end_date)
#     mask = df["city"].isin(sel_cities) & df["date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))
    
#     dff = df.loc[mask].copy()

#     st.title("🌤️ Painel Meteorológico")

#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Temp. média (°C)", f"{dff['temp_avg'].mean():.2f}")
#     with col2:
#         st.metric("Temp. máx (°C)", f"{dff['temp_max'].max():.2f}")
#     with col3:
#         st.metric("Umidade média (%)", f"{dff['rh_avg'].mean():.2f}")
#     with col4:
#         st.metric("Precipitação (mm)", f"{dff['precip_sum'].sum():.2f}")

#     st.subheader("Gráficos")
#     st.markdown('Por Temperatura Média')
#     st.line_chart(dff.pivot_table(index="date", columns="city", values="temp_avg").sort_index())
#     st.markdown('Por Umidade Relativa')
#     st.line_chart(dff.pivot_table(index="date", columns="city", values="rh_avg").sort_index())
#     # st.area_chart(dff.pivot_table(index="date", columns="city", values="precip_sum").sort_index())

#     st.subheader("Tabela")
#     st.dataframe(dff.reset_index(drop=True), use_container_width=True)

# except FileNotFoundError as e:
#     st.error(str(e))
