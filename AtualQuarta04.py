# app_ccc_mapas_era5_mm.py
# ----------------------------------------------------------
# Mapas ERA5 (dados reais, sem placeholders)
# - Fonte: Copernicus CDS • ERA5 Single Levels (Monthly means)
# - Variáveis: t2m (°C, média mensal) e tp (mm, acumulado mensal)
# - Área editável [N, W, E, S], export PNG/JPG/PDF/ZIP
# - Rodar com: streamlit run app_ccc_mapas_era5_mm.py
# ----------------------------------------------------------

import os
from io import BytesIO
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st
import cdsapi  # requer chave do CDS (arquivo ~/.cdsapirc local ou st.secrets no deploy)


# ===================== CONFIG UI =====================
st.set_page_config(page_title="CCC - ERA5 (Mapas)", page_icon="⛈️", layout="wide")
APP_TITLE = "CCC - Clima-Cast-Crepaldi"
APP_SUB   = "Mapas com dados reais do ERA5 (mensal)"

MESES = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
MES2NUM = {m: i+1 for i, m in enumerate(MESES)}
ANOS = list(range(1979, datetime.now().year + 1))   # ERA5 Single Levels começa em 1979

# Área padrão: América do Sul (CDS: [N, W, E, S])
AREA_DEFAULT = [15, -90, -30, -60]


# ===================== UTILS =====================
def _clean_filename(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+","_", s)
    s = s.replace("—","-").replace("–","-").replace("/","")
    s = re.sub(r"[^a-z0-9_\-\.]+","", s)
    return s

def horas_no_mes(ano: int, mes: int) -> int:
    per = pd.Period(f"{ano}-{mes:02d}")
    return per.days_in_month * 24

@st.cache_resource(show_spinner=False)
def cds_client():
    """Prioriza secrets em deploy; local cai no ~/.cdsapirc."""
    url = st.secrets.get("cds", {}).get("url", None)
    key = st.secrets.get("cds", {}).get("key", None)
    if url and key:
        return cdsapi.Client(url=url, key=key, verify=True)
    return cdsapi.Client()

@st.cache_data(show_spinner=True)
def download_era5_monthly(variable: str, year: int, month: int, area: list, grid: float = 0.25) -> bytes:
    """
    Faz request ao CDS e retorna NetCDF em bytes para ERA5 monthly means (single levels).
    variable: "2m_temperature" ou "total_precipitation"
    area: [North, West, East, South]
    grid: remapeamento (°); default 0.25
    """
    c = cds_client()
    dataset = "reanalysis-era5-single-levels-monthly-means"
    req = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": variable,
        "year": f"{year}",
        "month": f"{month:02d}",
        "time": "00:00",
        "area": area,                 # [N, W, E, S]
        "grid": [grid, grid],
        "format": "netcdf",
    }
    result = c.retrieve(dataset, req)
    return result.download_buffer().read()

def abrir_xarray(nc_bytes: bytes) -> xr.Dataset:
    return xr.open_dataset(BytesIO(nc_bytes))

def preparar_da(ds: xr.Dataset, var_label: str, ano: int, mes: int) -> xr.DataArray:
    """
    Converte para unidades finais:
      - Temperatura: t2m [K] → °C
      - Precipitação: tp monthly_mean [m] → total mensal [mm] = mean * horas_mes * 1000
    Renomeia dims latitude/longitude → lat/lon para plotar.
    """
    if var_label == "Temperatura":
        da = ds["t2m"] - 273.15
        da.attrs["units"] = "°C"
        da.attrs["long_name"] = "Temperatura média 2 m"
    else:
        mean_m = ds["tp"]                             # média mensal em metros
        total_m = mean_m * horas_no_mes(ano, mes)     # total do mês (m)
        da = total_m * 1000.0                         # mm
        da.attrs["units"] = "mm"
        da.attrs["long_name"] = "Precipitação acumulada (mensal)"

    # padroniza nomes
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da

def plot_da(da: xr.DataArray, titulo: str):
    """Plota com imshow usando extent lat/lon e colorbar de unidades."""
    fig, ax = plt.subplots(figsize=(9, 5))
    lat = da["lat"].values
    lon = da["lon"].values
    data = da.values

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    im = ax.imshow(
        data,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        aspect="auto",
    )
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(da.attrs.get("units", ""))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(titulo)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig

def fig_bytes(fig, fmt="png"):
    bio = BytesIO()
    fig.savefig(bio, format=fmt, dpi=160, bbox_inches="tight")
    return bio.getvalue()


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ERA5 — Configurações")
    st.caption(f"Agora: {datetime.now(ZoneInfo('America/Sao_Paulo')):%d/%m/%Y %H:%M}")

    var_label = st.selectbox("Variável:", ["Temperatura", "Precipitação"])
    year = st.select_slider("Ano:", options=ANOS, value=2021)
    month_name = st.radio("Mês:", MESES, horizontal=True, index=6)  # Jul
    month = MES2NUM[month_name]

    st.markdown("### Área [N, W, E, S] (CDS)")
    with st.expander("Editar área", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            north = st.number_input("Norte (lat)", value=float(AREA_DEFAULT[0]), step=1.0)
            west  = st.number_input("Oeste (lon)", value=float(AREA_DEFAULT[1]), step=1.0)
        with c2:
            east  = st.number_input("Leste (lon)", value=float(AREA_DEFAULT[2]), step=1.0)
            south = st.number_input("Sul (lat)",   value=float(AREA_DEFAULT[3]), step=1.0)
        area = [float(north), float(west), float(east), float(south)]

    st.markdown("### Exportar")
    nome_auto = _clean_filename(f"era5_{var_label.lower()}_mensal_{year}-{month:02d}")
    nome_atual = st.session_state.get("map_export_name", nome_auto)
    st.text_input("Nome do arquivo:", value=nome_atual, key="map_export_name")


# ===================== HEADER =====================
st.markdown(f"## {APP_TITLE}")
st.caption(APP_SUB)
st.markdown("---")


# ===================== CORPO: MAPA (dados reais) =====================
st.markdown("#### Resumo (Mapas)")
st.markdown(
    f"""
    <div style="display:flex;gap:.5rem;flex-wrap:wrap">
      <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Base: ERA5 (CDS)</span>
      <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Variável: {var_label}</span>
      <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Agregado: Mensal</span>
      <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Período: {year}-{month:02d}</span>
      <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Área: N{area[0]}, W{area[1]}, E{area[2]}, S{area[3]}</span>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    with st.spinner("Baixando ERA5 mensal do CDS…"):
        variable = "2m_temperature" if var_label == "Temperatura" else "total_precipitation"
        nc_bytes = download_era5_monthly(variable, year, month, area)
        ds = abrir_xarray(nc_bytes)
        da = preparar_da(ds, var_label, year, month)

    titulo = f"{da.attrs.get('long_name', var_label)} • {year}-{month:02d}"
    fig = plot_da(da, titulo)
    st.pyplot(fig, use_container_width=True)

    # ---- Downloads
    png = fig_bytes(fig, "png")
    jpg = fig_bytes(fig, "jpg")
    pdf = fig_bytes(fig, "pdf")
    map_name = st.session_state.get("map_export_name", nome_auto)

    zipb = BytesIO()
    with zipfile.ZipFile(zipb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{map_name}.png", png)
        zf.writestr(f"{map_name}.jpg", jpg)
        zf.writestr(f"{map_name}.pdf", pdf)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.download_button("⬇️ PNG", data=png, file_name=f"{map_name}.png", mime="image/png", use_container_width=True)
    with c2: st.download_button("⬇️ JPG", data=jpg, file_name=f"{map_name}.jpg", mime="image/jpeg", use_container_width=True)
    with c3: st.download_button("⬇️ PDF", data=pdf, file_name=f"{map_name}.pdf", mime="application/pdf", use_container_width=True)
    with c4: st.download_button("⬇️ ZIP", data=zipb.getvalue(), file_name=f"{map_name}.zip", mime="application/zip", use_container_width=True)

except Exception as e:
    st.error(f"Falha ao recuperar/plotar ERA5: {e}")
    st.stop()


# ===================== RODAPÉ (links úteis) =====================
with st.expander("Referências (ERA5, CDS)", expanded=False):
    st.markdown("""
- ERA5 Single Levels (Monthly means) — CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
- ERA5: documentação de variáveis e unidades (ECMWF/CKB): https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
- Como usar a API do CDS: https://cds.climate.copernicus.eu/how-to-api
""")
