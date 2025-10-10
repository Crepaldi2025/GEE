# ==================================================================================
#                         CCC - Clima-Cast-Crepaldi
# ==================================================================================
# Script Streamlit com integração ao CDSAPI (ERA5-Land) e Google Earth Engine
# Geração de mapas estáticos e interativos, além de séries temporais
# ==================================================================================

# ===================== CONFIG GERAL =====================
import os
from io import BytesIO
import zipfile
import tempfile
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st
import cdsapi
import math
import unicodedata, re
import time
from pathlib import Path

# Google Earth Engine
import ee

# Folium (mapas interativos)
from streamlit_folium import st_folium
import folium
import folium.plugins

# Mapa estático (plot de mapas)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd

import textwrap

# ===================== INICIALIZAÇÕES =====================
import ee
import streamlit as st
import json

import ee

# 1. Defina o caminho para o arquivo JSON de credenciais
SERVICE_ACCOUNT_FILE = 'C:\Users\crepa\Desktop\git\GEE\gee-crepaldi-2025-c050c2340f8e.json'

try:
    # 2. Leia e inicialize as credenciais
    ee.Initialize(
        credentials=ee.ServiceAccountCredentials(
            service_account_file=SERVICE_ACCOUNT_FILE,
            key_acct=None,  # 'key_acct' não é necessário quando key_file é fornecido
            project=None    # 'project' será lido do arquivo JSON
        )
    )
    print("Earth Engine inicializado com sucesso!")
except Exception as e:
    print(f"Erro ao inicializar o Earth Engine: {e}")


st.set_page_config(page_title="CCC - Clima-Cast-Crepaldi", page_icon="⛈️", layout="wide")
APP_TITLE = "CCC - Clima-Cast-Crepaldi"
PAGINAS = {
    "mapas": "Mapas",
    "series": "Séries Temporais",
    "sobre": "Sobre o Sistema",
}
# ==================================================================================
# GERENCIAMENTO DE ESTADO DA SESSÃO
# ==================================================================================
defaults = {
    "page": "home",
    "area_poligono_mapas": None,
    "area_poligono_series": None,
    "mapas_params": None,
    "series_params": None,
    "series_df": None,
    "series_df_multi": None,
    "series_how": "Média",
    "mapa_interativo": None,   # usado para Folium.Map
    "mapa_estatico": None,     # usado para Matplotlib fig
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================================================================================
# CONSTANTES
# ==================================================================================
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
MES2NUM = {m: i + 1 for i, m in enumerate(MESES)}
ANOS = list(range(1940, datetime.now().year + 1))

VAR_MAP_CDS = {
    "Temperatura": "2m_temperature",
    "Precipitação": "total_precipitation",
    "Vento": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
}
# ==================================================================================
# FUNÇÕES UTILITÁRIAS
# ==================================================================================
def _clean_filename(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("—", "-").replace("–", "-").replace("/", "")
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    return s

def _normalize_string(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")

@st.cache_resource(show_spinner=False)
def cds_client():
    url = st.secrets.get("cds", {}).get("url", None)
    key = st.secrets.get("cds", {}).get("key", None)
    if url and key:
        return cdsapi.Client(url=url, key=key, verify=True)
    return cdsapi.Client()

def _result_to_bytes(result) -> bytes:
    if hasattr(result, "download_buffer"):
        return result.download_buffer().read()
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        result.download(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

def _circle_polyline(lat0: float, lon0: float, radius_km: float, n: int = 240):
    delta_lat = radius_km / 111.0
    denom = max(1e-6, 111.0 * math.cos(math.radians(lat0)))
    delta_lon_base = radius_km / denom
    pts = []
    for k in range(n + 1):
        t = 2.0 * math.pi * k / n
        lat = lat0 + delta_lat * math.sin(t)
        lon = lon0 + delta_lon_base * math.cos(t)
        pts.append([lat, lon])
    return pts

def get_area_from_radius(lat: float, lon: float, radius_km: float):
    delta_lat = radius_km / 111.0
    lon_degree_len = 111.0 * math.cos(math.radians(lat))
    delta_lon = 0 if lon_degree_len == 0 else radius_km / lon_degree_len
    north = lat + delta_lat
    south = lat - delta_lat
    east = lon + delta_lon
    west = lon - delta_lon
    return [north, west, south, east]

# ==================================================================================
# DOWNLOAD ERA5
# ==================================================================================
@st.cache_data(show_spinner=False, max_entries=10)


def download_era5_data(base_dados, modo_agregado, variable, data_params, area):
    """
    Download de dados ERA5-Land (Copernicus Climate Data Store).

    Modo:
      - Diário        → reanalysis-era5-land (horário)
      - Mensal / Anual → reanalysis-era5-land-monthly-means
      - Personalizado → reanalysis-era5-land (horário)

    Parâmetros:
      base_dados  : string (mantido por compatibilidade)
      modo_agregado : str ("Diário", "Mensal", "Anual", "Personalizado")
      variable    : str ou list[str]
      data_params : dict contendo ano/mês/dia ou datas
      area        : list [Norte, Oeste, Sul, Leste]
    """
    import pandas as pd
    from datetime import date

    c = cds_client()

    # ==========================================================
    # Base comum da requisição
    # ==========================================================
    req = {
        "variable": tuple(variable) if isinstance(variable, list) else [variable],
        "area": area,
        "format": "netcdf",
        "grid": [0.1, 0.1],
    }

    # ==========================================================
    # Modo Diário (dataset horário)
    # ==========================================================
    if modo_agregado == "Diário":
        dataset = "reanalysis-era5-land"
        y, m, d = data_params["year"], data_params["month"], data_params["day"]
        req.update({
            "product_type": "reanalysis",
            "year": [f"{y}"],
            "month": [f"{m:02d}"],
            "day": [f"{d:02d}"],
            "time": [f"{h:02d}:00" for h in range(24)],
        })

    # ==========================================================
    # Modo Mensal (dataset de médias mensais)
    # ==========================================================
    elif modo_agregado == "Mensal":
        dataset = "reanalysis-era5-land-monthly-means"
        y, m = data_params["year"], data_params["month"]
        req.update({
            "product_type": "monthly_averaged_reanalysis",  # valor aceito pelo CDS
            "year": [f"{y}"],
            "month": [f"{m:02d}"],
            "time": "00:00",
        })

    # ==========================================================
    # Modo Anual (dataset de médias mensais)
    # ==========================================================
    elif modo_agregado == "Anual":
        dataset = "reanalysis-era5-land-monthly-means"
        y = data_params["year"]
        req.update({
            "product_type": "monthly_averaged_reanalysis",
            "year": [f"{y}"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
        })

    # ==========================================================
    # Modo Personalizado (dataset horário)
    # ==========================================================
    else:
        dataset = "reanalysis-era5-land"
        di = data_params["data_inicio"]
        df_ = data_params["data_fim"]

        if di > df_:
            raise ValueError("A data inicial não pode ser posterior à data final.")

        # Cria faixas únicas de ano, mês e dia
        date_range = pd.date_range(di, df_, freq="D")
        req.update({
            "product_type": "reanalysis",
            "year": sorted({f"{d.year}" for d in date_range}),
            "month": sorted({f"{d.month:02d}" for d in date_range}),
            "day": sorted({f"{d.day:02d}" for d in date_range}),
            "time": [f"{h:02d}:00" for h in range(24)],
        })

    # ==========================================================
    # Envio da requisição ao CDSAPI
    # ==========================================================
    try:
        result = c.retrieve(dataset, req)
        data_bytes = _result_to_bytes(result)
        return data_bytes
    except Exception as e:
        raise RuntimeError(f"❌ Falha ao baixar dados ERA5-Land ({dataset}): {e}")









@st.cache_data(show_spinner=False, max_entries=10)
def download_era5_data_diario(base_dados, variable, data_params, area):
    """
    Download ERA5-Land (horário) para um intervalo arbitrário,
    pensado para séries que serão agregadas para DIÁRIO depois.
    NÃO mexe nos mapas.
    """
    c = cds_client()
    di, df_ = data_params["data_inicio"], data_params["data_fim"]
    if di > df_:
        raise ValueError("A data de início não pode ser posterior à data de fim.")

    # Gera listas de ano/mês/dia cobrindo o intervalo
    rng = pd.date_range(di, df_, freq="D")
    years  = sorted({f"{d.year}"      for d in rng})
    months = sorted({f"{d.month:02d}" for d in rng})
    days   = sorted({f"{d.day:02d}"   for d in rng})

    req = {
        "format": "netcdf",
        "product_type": "reanalysis",
        "variable": tuple(variable) if isinstance(variable, list) else variable,
        "grid": [0.1, 0.1],
        "area": area,                      # [N, W, S, E]
        "year": years,
        "month": months,
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],  # sempre horário
    }

    result = c.retrieve("reanalysis-era5-land", req)
    return _result_to_bytes(result)

# ==================================================================================
# FUNÇÕES PARA ABRIR E PROCESSAR NETCDF (XARRAY)
# ==================================================================================
def abrir_xarray(nc_bytes: bytes) -> xr.Dataset:
    """Abre um dataset xarray a partir dos bytes do CDS (ZIP ou NetCDF)."""
    try:
        if nc_bytes[:2] == b"PK":  # ZIP
            with zipfile.ZipFile(BytesIO(nc_bytes), "r") as zf:
                nc_filenames = [name for name in zf.namelist() if name.endswith(".nc")]
                if not nc_filenames:
                    raise FileNotFoundError("Nenhum arquivo .nc encontrado dentro do ZIP do CDS.")
                nc_filename = nc_filenames[0]
                nc_file_bytes = zf.read(nc_filename)
                buffer = BytesIO(nc_file_bytes)
        else:
            buffer = BytesIO(nc_bytes)

        for engine in ["h5netcdf", "netcdf4", "scipy"]:
            try:
                buffer.seek(0)
                ds = xr.open_dataset(buffer, engine=engine)
                return ds
            except Exception:
                continue

        raise ValueError("Nenhum engine conseguiu abrir o NetCDF.")
    except Exception as e:
        raise RuntimeError(f"Falha ao abrir NetCDF: {e}")

def _coalesce(ds: xr.Dataset, keys):
    for k in keys:
        if k in ds:
            return ds[k]
    raise KeyError(f"Nenhuma das variáveis {keys} encontrada. Disponíveis: {list(ds.data_vars)}")

def _ensure_time_axis(da: xr.DataArray) -> xr.DataArray:
    """Garante que o eixo temporal se chame 'time' e seja datetime64."""
    if "valid_time" in da.dims and "time" not in da.dims:
        da = da.rename({"valid_time": "time"})
    if "time" not in da.dims and "time" in da.coords:
        pass
    elif "time" not in da.dims and "valid_time" in da.coords:
        vt = pd.to_datetime(da["valid_time"].values)
        da = da.expand_dims({"time": vt}).transpose("time", ...) if "time" not in da.dims else da

    if "time" in da.coords:
        if not np.issubdtype(da["time"].dtype, np.datetime64):
            da = da.assign_coords(time=pd.to_datetime(da["time"].values))
    else:
        raise KeyError("Eixo temporal ausente: nem 'time' nem 'valid_time' encontrados.")
    return da

def _collapse_aux_dims(da: xr.DataArray) -> xr.DataArray:
    """Remove/colapsa dimensões auxiliares (expver/number)."""
    if "expver" in da.dims:
        da = da.max("expver", skipna=True)
    if "number" in da.dims:
        da = da.mean("number", skipna=True)
    drop_coords = [c for c in ("expver", "number") if c in da.coords and c not in da.dims]
    if drop_coords:
        da = da.reset_coords(drop=True)
    return da

def _subset_bbox(da: xr.DataArray, bbox):
    """Recorte espacial por bounding box [N, W, S, E] (ordem CDS)."""
    N, W, S, E = bbox
    lat_name = "latitude" if "latitude" in da.dims else ("lat" if "lat" in da.dims else None)
    lon_name = "longitude" if "longitude" in da.dims else ("lon" if "lon" in da.dims else None)
    if lat_name is None or lon_name is None:
        raise ValueError("Variável sem dimensões latitude/longitude reconhecidas.")
    lat_vals = da[lat_name]
    if float(lat_vals[0]) > float(lat_vals[-1]):  # lat decrescente (ERA5 usual)
        da = da.sel({lat_name: slice(N, S), lon_name: slice(W, E)})
    else:
        da = da.sel({lat_name: slice(S, N), lon_name: slice(W, E)})
    return da

def _temporal_reduce(da: xr.DataArray, var_label: str, freq: str) -> xr.DataArray:
    """Agregação temporal: precipitação soma; demais, média."""
    da = _ensure_time_axis(da)
    if var_label == "Precipitação":
        da_mm = da * 1000.0
        out = da_mm.resample(time=freq).sum(skipna=True)
        out.attrs["units"] = "mm"
    else:
        out = da.resample(time=freq).mean(skipna=True)
        out.attrs["units"] = "°C" if var_label == "Temperatura" else "m/s"
    return out

def _spatial_reduce(da: xr.DataArray, how: str) -> xr.DataArray:
    da = _collapse_aux_dims(da)
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    if how == "Mínimo":
        return da.min(dim=(lat_name, lon_name), skipna=True)
    elif how == "Máximo":
        return da.max(dim=(lat_name, lon_name), skipna=True)
    else:
        return da.mean(dim=(lat_name, lon_name), skipna=True)
def build_series(nc_bytes: bytes, var_label: str, bbox, freq: str, how: str) -> pd.DataFrame:
    ds = abrir_xarray(nc_bytes)
    if var_label == "Vento":
        u10 = _coalesce(ds, ["u10", "10m_u_component_of_wind"])
        v10 = _coalesce(ds, ["v10", "10m_v_component_of_wind"])
        da = np.sqrt(u10**2 + v10**2)
    elif var_label == "Temperatura":
        t2m = _coalesce(ds, ["t2m", "2m_temperature"])
        da = t2m - 273.15
    else:
        da = _coalesce(ds, ["tp", "total_precipitation", "total_precipitation_sum"])

    da = _ensure_time_axis(da)
    da = _collapse_aux_dims(da)
    da = _subset_bbox(da, bbox)
    if da.size == 0 or da.time.size == 0:
        raise ValueError("Nenhum dado encontrado para a área/período selecionados.")

    da_t = _temporal_reduce(da, var_label, freq=freq)
    da_s = _spatial_reduce(da_t, how)
    df = da_s.to_dataframe(name="valor").reset_index().set_index("time").sort_index()
    if df.empty:
        raise ValueError("Série resultou vazia após a agregação.")
    return df

# ==================================================================================
# FUNÇÕES DE PLOTAGEM
# ==================================================================================


def plot_da(da: xr.DataArray, titulo: str, var_label: str, geodf=None,
            lat_center=None, lon_center=None, raio_km=None):
    if da.size == 0:
        raise ValueError("Mapa vazio: nenhum dado retornado para a seleção.")

    # ✅ Detecta nomes das dimensões espaciais
    lat_name = "latitude" if "latitude" in da.dims else ("lat" if "lat" in da.dims else None)
    lon_name = "longitude" if "longitude" in da.dims else ("lon" if "lon" in da.dims else None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Não encontrei dimensões de latitude/longitude em {list(da.dims)}")

    # ✅ Remove/agrupa dimensões não-espaciais (ex.: time, expver, number)
    extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
    for d in extra_dims:
        if da.sizes.get(d, 1) == 1:
            da = da.isel({d: 0}, drop=True)  # elimina eixos singleton
        else:
            da = da.mean(dim=d, skipna=True)  # agrega se >1, garantindo 2D

    # ✅ Garante ordem (lat, lon) e que é 2D
    if set([lat_name, lon_name]).issubset(da.dims):
        da = da.transpose(lat_name, lon_name)
    else:
        raise ValueError(f"A matriz não está em {lat_name}×{lon_name} após ‘squeeze’. Dims atuais: {da.dims}")
    data_values = np.asarray(da.values)
    if data_values.ndim != 2:
        raise ValueError(f"Esperado 2D após ‘squeeze/mean’, recebi {data_values.shape}.")

    # ======= Figura base =======
    fig = plt.figure(figsize=(10, 8))
    lon_min, lon_max = float(da[lon_name].min()), float(da[lon_name].max())
    lat_min, lat_max = float(da[lat_name].min()), float(da[lat_name].max())
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray')

    lat_values = da[lat_name].values
    lon_values = da[lon_name].values

    # ✅ Garante orientação crescente da latitude para o imshow
    if lat_values[0] > lat_values[-1]:
        lat_values = lat_values[::-1]
        data_values = data_values[::-1, :]

    # Escala de cores
    vmin = float(np.nanmin(data_values))
    vmax = float(np.nanmax(data_values))
    if np.isclose(vmin, vmax):
        vmin, vmax = None, None
    cmap_map = {"Temperatura": "coolwarm", "Precipitação": "YlGnBu", "Vento": "YlOrRd"}

    im = ax.imshow(
        data_values,
        transform=ccrs.PlateCarree(),
        extent=[lon_values.min(), lon_values.max(), lat_values.min(), lat_values.max()],
        cmap=cmap_map.get(var_label, "viridis"),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # Contorno (opcional)
    if geodf is not None:
        ax.add_geometries(geodf.geometry, crs=ccrs.PlateCarree(),
                          edgecolor='dimgray', facecolor='none', linewidth=1.2)

    # Grade/labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Círculo (opcional)
    if lat_center is not None and lon_center is not None and raio_km is not None:
        pts = _circle_polyline(lat_center, lon_center, raio_km)
        lats_c, lons_c = zip(*pts)
        ax.plot(lons_c, lats_c, transform=ccrs.PlateCarree(),
                color='white', linestyle='--', linewidth=2.5, zorder=5)
        ax.scatter([lon_center], [lat_center], transform=ccrs.PlateCarree(),
                   color='white', edgecolor='black', s=80, zorder=6)

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.05)
    cb.set_label(da.attrs.get("units", "Unidade"))
    ax.set_title(titulo)
    fig.tight_layout()
    return fig





def plot_series_multi(df_multi: pd.DataFrame, titulo: str, unidade: str, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n = len(df_multi.index)
    use_marker = n <= 2
    if "minimo" in df_multi:
        ax.plot(df_multi.index, df_multi["minimo"], label="Mínimo", marker='o' if use_marker else None)
    if "media" in df_multi:
        ax.plot(df_multi.index, df_multi["media"], label="Média", marker='o' if use_marker else None)
    if "maximo" in df_multi:
        ax.plot(df_multi.index, df_multi["maximo"], label="Máximo", marker='o' if use_marker else None)
    if xlim is not None:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    elif n == 1:
        t = pd.to_datetime(df_multi.index[0])
        pad = pd.Timedelta(days=15)
        ax.set_xlim(t - pad, t + pad)
    ax.set_title(titulo)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(unidade)
    ax.grid(True, alpha=.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_series(df: pd.DataFrame, titulo: str, unidade: str, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n = len(df.index)

    # marcador só se poucos pontos
    use_marker = n <= 50  

    ax.plot(df.index, df["valor"], label="Média", marker='o' if use_marker else None)

    if xlim is not None:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    elif n == 1:
        t = pd.to_datetime(df.index[0])
        pad = pd.Timedelta(days=15)
        ax.set_xlim(t - pad, t + pad)

    ax.set_title(titulo)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(unidade)
    ax.grid(True, alpha=.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt="png"):
    bio = BytesIO()
    fig.savefig(bio, format=fmt, dpi=160, bbox_inches="tight")
    return bio.getvalue()

# ==================================================================================
# INTEGRAÇÃO COM GOOGLE EARTH ENGINE (GEE)
# ==================================================================================
@st.cache_data(show_spinner=False)
def get_gee_collection_states():
    """Carrega apenas os contornos de estados (ADM1, sem municípios)."""
    return ee.FeatureCollection("FAO/GAUL/2015/level1")

@st.cache_data(show_spinner=False)
def get_gee_collection():
    """Carrega contornos de municípios (ADM2)."""
    return ee.FeatureCollection("FAO/GAUL/2015/level2")

@st.cache_data(show_spinner=False)
def get_geometry_gee(uf_name=None, mun_name=None):
    """Retorna geometria ee.Geometry para recorte no GEE (não usar no Folium)."""
    collection = ee.FeatureCollection("FAO/GAUL/2015/level2")
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    gee_mun_name = _normalize_string(mun_name) if mun_name else None

    if uf_name and not mun_name:
        feature = collection.filter(ee.Filter.eq("ADM1_NAME", gee_uf_name))
    elif uf_name and mun_name:
        feature = collection.filter(
            ee.Filter.And(
                ee.Filter.eq("ADM1_NAME", gee_uf_name),
                ee.Filter.eq("ADM2_NAME", gee_mun_name),
            )
        )
    else:
        feature = collection.filter(ee.Filter.eq("ADM0_NAME", "Brazil"))

    return feature.geometry()

@st.cache_data(show_spinner=False)
def get_municipio_polygon_gee(_collection, uf_name, mun_name):
    """GeoDataFrame para MUNICÍPIO (ADM2) a partir do GAUL/2015/level2. Usado no Folium."""
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    gee_mun_name = _normalize_string(mun_name)
    ee_feature = _collection.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM1_NAME", gee_uf_name),
            ee.Filter.eq("ADM2_NAME", gee_mun_name),
        )
    )
    json_geom = ee_feature.geometry().getInfo()
    return gpd.GeoDataFrame.from_features(
        [{"geometry": json_geom, "properties": {}}], crs="EPSG:4326"
    )
# ==================================================================================
# MAPAS DE NOMES (Português ↔ GEE)
# ==================================================================================
_UF_PT_TO_GEE = {
    "Amapá": "Amapa", "Ceará": "Ceara", "Espírito Santo": "Espirito Santo", "Goiás": "Goias",
    "Maranhão": "Maranhao", "Paraíba": "Paraiba", "Paraná": "Parana", "Piauí": "Piaui",
    "Rondônia": "Rondonia", "São Paulo": "Sao Paulo", "Pará": "Para", "Rio Grande do Sul": "Rio Grande do Sul",
    "Minas Gerais": "Minas Gerais", "Mato Grosso do Sul": "Mato Grosso do Sul", "Mato Grosso": "Mato Grosso",
    "Rio Grande do Norte": "Rio Grande do Norte", "Rio de Janeiro": "Rio de Janeiro",
    "Santa Catarina": "Santa Catarina", "Bahia": "Bahia", "Alagoas": "Alagoas",
    "Sergipe": "Sergipe", "Pernambuco": "Pernambuco", "Tocantins": "Tocantins",
    "Acre": "Acre", "Amazonas": "Amazonas", "Roraima": "Roraima", "Distrito Federal": "Distrito Federal",
}

_GEE_TO_PT_FIX = {
    "Amapa": "Amapá", "Ceara": "Ceará", "Espirito Santo": "Espírito Santo", "Goias": "Goiás",
    "Maranhao": "Maranhão", "Paraiba": "Paraíba", "Parana": "Paraná", "Piaui": "Piauí",
    "Rondonia": "Rondônia", "Sao Paulo": "São Paulo", "Para": "Pará",
}


@st.cache_data(show_spinner=False)
def get_uf_names(_collection):
    """Retorna lista de nomes de estados (UFs) do Brasil, sem duplicatas ou valores inválidos."""
    brazil_regions = _collection.filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
    uf_list_gee = (
        brazil_regions.aggregate_array("ADM1_NAME")
        .distinct()
        .sort()
        .getInfo()
    )

    # 🔧 Remove entradas inválidas ou desconhecidas
    uf_list_gee = [
        name for name in uf_list_gee
        if name and isinstance(name, str)
        and name.lower().strip() not in ["", "name unknown", "unknown", "nan"]
    ]

    # Converte para nomes em português corrigidos
    uf_list_pt = sorted([_GEE_TO_PT_FIX.get(name, name) for name in uf_list_gee])
    return uf_list_pt





@st.cache_data(show_spinner=False)
def get_state_polygon_gee(_collection, uf_name):
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    ee_feature = _collection.filter(ee.Filter.eq("ADM1_NAME", gee_uf_name))
    json_geom = ee_feature.geometry().getInfo()
    return gpd.GeoDataFrame.from_features(
        [{"geometry": json_geom, "properties": {}}], crs="EPSG:4326"
    )

@st.cache_data(show_spinner=False)
def get_mun_names(_collection, uf_name):
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    mun_list = (
        _collection.filter(
            ee.Filter.And(
                ee.Filter.eq("ADM0_NAME", "Brazil"),
                ee.Filter.eq("ADM1_NAME", gee_uf_name),
            )
        )
        .aggregate_array("ADM2_NAME")
        .distinct()
        .sort()
        .getInfo()
    )
    return sorted(mun_list)

def get_area_from_gee(_collection, uf_name=None, mun_name=None):
    """Retorna bounding box [N, W, S, E] de um estado/município a partir do GEE."""
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    gee_mun_name = _normalize_string(mun_name) if mun_name else None

    if uf_name and not mun_name:
        feature = _collection.filter(ee.Filter.eq("ADM1_NAME", gee_uf_name)).geometry()
    elif uf_name and mun_name:
        feature = _collection.filter(
            ee.Filter.And(
                ee.Filter.eq("ADM1_NAME", gee_uf_name),
                ee.Filter.eq("ADM2_NAME", gee_mun_name),
            )
        ).geometry()
    else:
        feature = _collection.filter(ee.Filter.eq("ADM0_NAME", "Brazil")).geometry()

    bounds = feature.bounds().getInfo()["coordinates"][0]
    lat_max = bounds[2][1]
    lon_min = bounds[0][0]
    lat_min = bounds[0][1]
    lon_max = bounds[2][0]
    return [lat_max, lon_min, lat_min, lon_max]

# ==================================================================================
# UI COMPARTILHADA (SIDEBAR)
# ==================================================================================
def ui_sidebar_escolhas(prefix: str):
    st.header("Menu (Mapas)")
    st.caption(f"Agora: {datetime.now(ZoneInfo('America/Sao_Paulo')):%d/%m/%Y %H:%M}")
    st.sidebar.markdown("---")

    st.markdown("### Base de Dados")
    base_dados = st.selectbox("Base de Dados", ["ERA5-LAND"], key=f"{prefix}_base", label_visibility="hidden")
    st.sidebar.markdown("---")

    st.markdown("### Variável")
    var_label = st.selectbox("Variável", ["Precipitação", "Temperatura", "Vento"],
                             key=f"{prefix}_var", label_visibility="hidden")

    st.markdown("### Agregado temporal")
    modo_agregado = st.selectbox("Agregado temporal", ["Mensal", "Anual", "Personalizado"],
                                 key=f"{prefix}_agregado", label_visibility="hidden")
    st.sidebar.markdown("---")

    st.markdown("### Período")
    periodo_str, data_params, freq_code = "", {}, None
    if modo_agregado == "Mensal":
        year = st.select_slider("Ano:", options=ANOS, value=2021, key=f"{prefix}_ano_m")
        month_name = st.radio("Mês:", MESES, horizontal=True, index=6, key=f"{prefix}_mes_m")
        month = MES2NUM[month_name]
        periodo_str = f"{year}/{month:02d}"
        data_params = {"year": year, "month": month}
    elif modo_agregado == "Anual":
        year = st.select_slider("Ano:", options=ANOS, value=2021, key=f"{prefix}_ano_a")
        periodo_str = f"{year}"
        data_params = {"year": year}
    else:
        di = st.date_input("Data de Início (YYYY/MM/DD):", value=date(2021, 1, 1), key=f"{prefix}_inicio_p")
        df_ = st.date_input("Data de Fim (YYYY/MM/DD):", value=date(2021, 1, 31), key=f"{prefix}_fim_p")
        periodo_str = f"de {di.strftime('%d/%m/%Y')} a {df_.strftime('%d/%m/%Y')}"
        freq_code = "D"
        data_params = {"data_inicio": di, "data_fim": df_, "freq_code": freq_code}
    st.sidebar.markdown("---")
    st.markdown("### Área de Interesse")
    modo_area = st.radio("Modo de seleção:", ["Estado", "Município", "Polígono Personalizado", "Círculo (Lat, Lon, Raio)"],
                         key=f"{prefix}_modo_area")
    area_params = {"modo_area": modo_area}
    collection = get_gee_collection()

    if modo_area == "Estado":
        uf_list = get_uf_names(collection)
        selected_uf = st.selectbox("Selecione o Estado:", uf_list, key=f"{prefix}_estado")
        area_params.update({"selected_uf": selected_uf, "area_label": f"Estado de {selected_uf}"})
    elif modo_area == "Município":
        uf_list = get_uf_names(collection)
        selected_uf = st.selectbox("Selecione o Estado:", uf_list, key=f"{prefix}_estado2")
        area_params["selected_uf"] = selected_uf
        if selected_uf:
            mun_list = get_mun_names(collection, selected_uf)
            selected_mun = st.selectbox("Selecione o Município:", mun_list, key=f"{prefix}_mun")
            area_params.update({"selected_mun": selected_mun, "area_label": f"Município de {selected_mun}/{selected_uf}"})
    elif modo_area == "Polígono Personalizado":
        area_params["area_label"] = "Polígono Personalizado"
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        lat_center = st.number_input("Latitude:", value=-22.0, format="%.4f", key=f"{prefix}_lat")
        lon_center = st.number_input("Longitude:", value=-46.0, format="%.4f", key=f"{prefix}_lon")
        raio_km = st.number_input("Raio (km):", value=100.0, min_value=1.0, format="%.2f", key=f"{prefix}_raio")
        area_params.update({"lat_center": lat_center, "lon_center": lon_center, "raio_km": raio_km,
                            "area_label": f"Círculo (Lat: {lat_center}, Lon: {lon_center}, Raio: {raio_km} km)"})

    st.sidebar.markdown("---")
    with st.form(f"{prefix}_form"):
        submitted = st.form_submit_button("Revisar Parâmetros", use_container_width=True, type="primary")

    params = {
        "base_dados": base_dados,
        "var_label": var_label,
        "modo_agregado": modo_agregado,
        "periodo_str": periodo_str,
        "data_params": data_params,
        "area_params": area_params,
    }
    if freq_code is not None:
        params["data_params"]["freq_code"] = freq_code
    return submitted, params

def ui_sidebar_escolhas_series(prefix: str):
    st.header("Menu (Séries Temporais)")
    st.caption(f"Agora: {datetime.now(ZoneInfo('America/Sao_Paulo')):%d/%m/%Y %H:%M}")
    st.sidebar.markdown("---")
    st.markdown("### Base de Dados")
    base_dados = st.selectbox("Base de Dados", ["ERA5-LAND"], key=f"{prefix}_base", label_visibility="hidden")
    st.sidebar.markdown("---")

    st.markdown("### Variável")
    var_label = st.selectbox("Variável", ["Precipitação", "Temperatura", "Vento"],
                             key=f"{prefix}_var", label_visibility="hidden")
    st.sidebar.markdown("---")

    st.markdown("### Período")
    di = st.date_input("Data de Início (YYYY/MM/DD):", value=date(2021, 1, 1), key=f"{prefix}_inicio")
    df_ = st.date_input("Data de Fim (YYYY/MM/DD):", value=date(2021, 1, 31), key=f"{prefix}_fim")
    periodo_str = f"de {di.strftime('%d/%m/%Y')} a {df_.strftime('%d/%m/%Y')}"
    data_params = {"data_inicio": di, "data_fim": df_, "freq_code": "D"}
    st.sidebar.markdown("---")
    st.markdown("### Área de Interesse")
    modo_area = st.radio("Modo de seleção:", ["Estado", "Município", "Polígono Personalizado", "Círculo (Lat, Lon, Raio)"],
                         key=f"{prefix}_modo_area")
    area_params = {"modo_area": modo_area}
    collection = get_gee_collection()

    if modo_area == "Estado":
        uf_list = get_uf_names(collection)
        selected_uf = st.selectbox("Selecione o Estado:", uf_list, key=f"{prefix}_estado")
        area_params.update({"selected_uf": selected_uf, "area_label": f"Estado de {selected_uf}"})
    elif modo_area == "Município":
        uf_list = get_uf_names(collection)
        selected_uf = st.selectbox("Selecione o Estado:", uf_list, key=f"{prefix}_estado2")
        area_params["selected_uf"] = selected_uf
        if selected_uf:
            mun_list = get_mun_names(collection, selected_uf)
            selected_mun = st.selectbox("Selecione o Município:", mun_list, key=f"{prefix}_mun")
            area_params.update({"selected_mun": selected_mun, "area_label": f"Município de {selected_mun}/{selected_uf}"})
    elif modo_area == "Polígono Personalizado":
        area_params["area_label"] = "Polígono Personalizado"
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        lat_center = st.number_input("Latitude:", value=-22.0, format="%.4f", key=f"{prefix}_lat")
        lon_center = st.number_input("Longitude:", value=-46.0, format="%.4f", key=f"{prefix}_lon")
        raio_km = st.number_input("Raio (km):", value=100.0, min_value=1.0, format="%.2f", key=f"{prefix}_raio")
        area_params.update({"lat_center": lat_center, "lon_center": lon_center, "raio_km": raio_km,
                            "area_label": f"Círculo (Lat: {lat_center}, Lon: {lon_center}, Raio: {raio_km} km)"})
    st.sidebar.markdown("---")
    with st.form(f"{prefix}_form"):
        submitted = st.form_submit_button("Revisar Parâmetros", use_container_width=True, type="primary")

    params = {
        "base_dados": base_dados,
        "var_label": var_label,
        "periodo_str": periodo_str,
        "data_params": data_params,
        "area_params": area_params,
    }
    return submitted, params

# ==================================================================================
# UI PREVIEW DE ÁREA (polígono ou círculo)
# ==================================================================================

def ui_previews(params: dict, prefix: str):
    modo_area = params['area_params']['modo_area']
    if modo_area == "Polígono Personalizado":
        st.subheader("Desenhe seu Polígono no Mapa")
        m = folium.Map(location=[-14.235, -51.9253], zoom_start=4, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Satellite",
            name="Esri Satellite",
            overlay=False,
            control=True
        ).add_to(m)

        draw = folium.plugins.Draw(
            position='topleft',
            draw_options={
                'polyline': False, 'rectangle': False, 'circle': False,
                'circlemarker': False, 'marker': False, 'polygon': {'showArea': True}
            },
            edit_options={'edit': False, 'remove': True}
        )
        m.add_child(draw)
        out = st_folium(m, width=800, height=520, key=f"preview_{prefix}", returned_objects=['all_drawings'])
        if out and out.get("all_drawings"):
            poly_data = out["all_drawings"][0]
            if poly_data.get('geometry', {}).get('type', '').lower() == 'polygon':
                coords = poly_data['geometry']['coordinates'][0]
                lats = [c[1] for c in coords]
                lons = [c[0] for c in coords]
                st.session_state[f"area_poligono_{prefix}"] = [max(lats), min(lons), min(lats), max(lons)]

    elif modo_area == "Círculo (Lat, Lon, Raio)":
        st.subheader("Pré-visualização da Área Circular")
        lat0 = params['area_params'].get("lat_center", -14.235)
        lon0 = params['area_params'].get("lon_center", -51.9253)
        rkm = params['area_params'].get("raio_km", 100.0)
        m = folium.Map(location=[lat0, lon0], zoom_start=6, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Satellite",
            name="Esri Satellite",
            overlay=False,
            control=True
        ).add_to(m)

        pts = _circle_polyline(lat0, lon0, rkm, n=240)
        folium.PolyLine(locations=pts, color="white", weight=3, opacity=1.0, dash_array="6,8").add_to(m)
        folium.CircleMarker(
            location=[lat0, lon0], radius=6, color="white", weight=3,
            fill=True, fill_color="white", fill_opacity=1.0, tooltip="Centro"
        ).add_to(m)
        north, west, south, east = get_area_from_radius(lat0, lon0, rkm)
        m.fit_bounds([[south, west], [north, east]])
        st_folium(m, width=800, height=520, key=f"preview_{prefix}")

# ==================================================================================
# UI REVISÃO DE PARÂMETROS
# ==================================================================================

def ui_revisao(params: dict) -> bool:
    st.markdown("#### Revisão dos Parâmetros")

    # ✅ Validação de período apenas se for personalizado
    if "modo_agregado" in params and params["modo_agregado"] == "Personalizado":
        di, df_ = params["data_params"]["data_inicio"], params["data_params"]["data_fim"]
        if di > df_:
            st.error("Erro: A data de início não pode ser posterior à data de fim.")
            return False
    elif "modo_agregado" not in params:
        # Séries temporais (sem agregado)
        di, df_ = params["data_params"]["data_inicio"], params["data_params"]["data_fim"]
        if di > df_:
            st.error("Erro: A data de início não pode ser posterior à data de fim.")
            return False

    # ✅ Montagem dos chips de revisão
    chips = [
        f"Base: {params['base_dados']}",
        f"Variável: {params['var_label']}",
        f"Período: {params['periodo_str']}",
        f"Área: {params['area_params']['area_label']}",
    ]

    if "modo_agregado" in params:
        chips.insert(2, f"Agregado: {params['modo_agregado']}")

    st.markdown(
        "<div style='display:flex;gap:.5rem;flex-wrap:wrap'>" +
        "".join([f"<span style='padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px'>{c}</span>" for c in chips]) +
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    return True

# ==================================================================================
# AVISO DIDÁTICO DE DESEMPENHO
# ==================================================================================
def _estimate_bbox_area_km2(area_bbox):
    """Estimativa rápida da área (km²) do bounding box [N, W, S, E]."""
    if not area_bbox:
        return None
    N, W, S, E = area_bbox
    lat_mid = (N + S) / 2.0
    dy_km = abs(N - S) * 111.0
    dx_km = abs(E - W) * 111.0 * max(1e-6, math.cos(math.radians(lat_mid)))
    return max(0.0, dx_km * dy_km)

def _count_hours(modo_agregado, data_params):
    """Quantifica a carga temporal que será baixada do CDS/GEE (em horas)."""
    if modo_agregado == "Mensal":
        y, m = data_params["year"], data_params["month"]
        days = pd.Period(f"{y}-{m:02d}").days_in_month
        return 24 * days
    elif modo_agregado == "Anual":
        # ERA5-Land monthly means → 12 valores mensais (não-horário)
        return 12  # marcador didático
    else:  # Personalizado (horário)
        di, df_ = data_params["data_inicio"], data_params["data_fim"]
        # inclui a última data no cálculo de horas
        nh = pd.date_range(di, df_ + pd.Timedelta(days=1), freq="H", inclusive="left").size
        return int(nh)

def show_performance_hint(params: dict, etapa: str, area_bbox=None):
    st.warning(
        "⚠️ Dependendo da escolha dos parâmetros a consulta pode levar alguns minutos."
    )

# ==================================================================================
# FUNÇÃO DE EXPORTAÇÃO
# ==================================================================================
def export_buttons(file_name: str, csv_data: bytes, excel_buffer: BytesIO, plot_fig=None):
    st.markdown("### Exportar")
    if plot_fig:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Imagens")
            png = fig_bytes(plot_fig, "png")
            jpg = fig_bytes(plot_fig, "jpg")
            pdf = fig_bytes(plot_fig, "pdf")
            zipb = BytesIO()
            with zipfile.ZipFile(zipb, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{file_name}.png", png)
                zf.writestr(f"{file_name}.jpg", jpg)
                zf.writestr(f"{file_name}.pdf", pdf)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button("⬇️ PNG", data=png, file_name=f"{file_name}.png",
                                   mime="image/png", use_container_width=True)
            with c2:
                st.download_button("⬇️ JPG", data=jpg, file_name=f"{file_name}.jpg",
                                   mime="image/jpeg", use_container_width=True)
            with c3:
                st.download_button("⬇️ PDF", data=pdf, file_name=f"{file_name}.pdf",
                                   mime="application/pdf", use_container_width=True)
            with c4:
                st.download_button("⬇️ ZIP", data=zipb.getvalue(), file_name=f"{file_name}.zip",
                                   mime="application/zip", use_container_width=True)
        with col2:
            st.markdown("#### Dados")
            c1d, c2d = st.columns(2)
            with c1d:
                st.download_button("⬇️ CSV", data=csv_data, file_name=f"{file_name}.csv",
                                   mime="text/csv", use_container_width=True)
            with c2d:
                st.download_button("⬇️ XLSX", data=excel_buffer, file_name=f"{file_name}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
    else:
        st.markdown("---")
        c1d, c2d = st.columns(2)
        with c1d:
            st.download_button("⬇️ CSV", data=csv_data, file_name=f"{file_name}.csv",
                               mime="text/csv", use_container_width=True)
        with c2d:
            st.download_button("⬇️ XLSX", data=excel_buffer, file_name=f"{file_name}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

def gerar_mapa_simplificado(uf_name="Minas Gerais"):
    # Carrega apenas estados (GAUL nível 1)
    collection = get_gee_collection_states()
    geom = collection.filter(ee.Filter.eq("ADM1_NAME", _UF_PT_TO_GEE.get(uf_name, uf_name))).geometry()

    # Bounding box do estado
    bounds = geom.bounds().getInfo()["coordinates"][0]
    lat_max = bounds[2][1]
    lon_min = bounds[0][0]
    lat_min = bounds[0][1]
    lon_max = bounds[2][0]
    center_lat, center_lon = (lat_max + lat_min) / 2, (lon_min + lon_max) / 2

    # Cria um mapa Folium simples
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Adiciona apenas o contorno do estado
    folium.GeoJson(
        data=geom.getInfo(),
        style_function=lambda x: {"fillColor": "none", "color": "blue", "weight": 2}
    ).add_to(m)

    return m

def tela_mapas():

    st.markdown("---")
    st.markdown(f"## {PAGINAS['mapas']}")
    st.markdown("Nesta página você pode gerar **mapas** com dados climáticos.")

    # ===================== Resultado atual (PERSISTENTE) =====================
    st.markdown("### Resultado atual")
    mostrou_alguma_coisa = False
    if st.button("🗑️ Limpar resultado atual (mapas)", type="secondary", key="limpar_mapas"):
        st.session_state["mapa_interativo_html"] = None
        st.session_state["mapa_estatico"] = None
        st.session_state["mapa_estatico_dados"] = None
        st.session_state["mapas_params"] = None
        st.success("Resultados de mapas apagados. Faça uma nova escolha na sidebar.")
        st.rerun()

    # Exibir mapa interativo salvo (HTML)
    if "mapa_interativo_html" in st.session_state and st.session_state["mapa_interativo_html"]:
        st.components.v1.html(
            st.session_state["mapa_interativo_html"],
            height=600,
            scrolling=True
        )
        mostrou_alguma_coisa = True

    # Exibir mapa estático salvo
    if st.session_state.get("mapa_estatico") is not None:
        st.pyplot(st.session_state["mapa_estatico"])
        mostrou_alguma_coisa = True

        # ✅ Exportação e tabela para o mapa estático
        if st.session_state.get("mapa_estatico_dados") is not None and st.session_state.get("mapas_params") is not None:
            params = st.session_state["mapas_params"]
            df_export = st.session_state["mapa_estatico_dados"]

            nome_auto = _clean_filename(
                f"era5_{params['var_label'].lower()}_{params['modo_agregado'].replace(' ', '_').lower()}_{params['periodo_str'].replace('/', '-')}"
            )
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Dados')
            excel_buffer.seek(0)

            export_buttons(nome_auto, csv_data, excel_buffer, plot_fig=st.session_state["mapa_estatico"])

            with st.expander("Mostrar dados em tabela"):
                st.dataframe(df_export, use_container_width=True)

    if not mostrou_alguma_coisa:
        st.caption("Ainda não há mapas gerados nesta sessão.")

    # ===================== Sidebar de parâmetros =====================
    with st.sidebar:
        submitted, params = ui_sidebar_escolhas(prefix="mapas")

    if submitted:
        st.session_state.mapas_params = params

    if not st.session_state.mapas_params:
        ui_previews(params, prefix="mapas")
        st.info("Selecione os parâmetros e clique em **Revisar Parâmetros**.")
        return

    params = st.session_state.mapas_params
    if not ui_revisao(params):
        return

    # ===================== Determinar área =====================
    modo_area = params["area_params"]["modo_area"]
    collection = get_gee_collection()
    area, geodf = None, None

    if modo_area == "Estado":
        uf = params["area_params"]["selected_uf"]
        geodf = get_state_polygon_gee(get_gee_collection_states(), uf)
        area = get_area_from_gee(collection, uf_name=uf)

    elif modo_area == "Município":
        uf, mun = params["area_params"]["selected_uf"], params["area_params"]["selected_mun"]
        geodf = get_municipio_polygon_gee(collection, uf, mun)
        area = get_area_from_gee(collection, uf_name=uf, mun_name=mun)

    elif modo_area == "Círculo (Lat, Lon, Raio)":
        lat0 = params["area_params"]["lat_center"]
        lon0 = params["area_params"]["lon_center"]
        rkm  = params["area_params"]["raio_km"]
        north, west, south, east = get_area_from_radius(lat0, lon0, rkm)
        area = [north, west, south, east]

    elif modo_area == "Polígono Personalizado":
        area = st.session_state.area_poligono_mapas
        if area is None:
            st.warning("⚠️ Desenhe o polígono no mapa acima para continuar.")
            ui_previews(params, "mapas")
            return

    if area is None:
        st.error("❌ Não foi possível determinar a área de interesse.")
        return

    params["area"] = area



    

    # ===================== Aviso de desempenho =====================
    show_performance_hint(params, etapa="mapas", area_bbox=area)
    st.subheader("Gerar Mapas")

    # ===================== Botões =====================
    col1, col2 = st.columns(2)
    with col1:
        gerar_interativo = st.button("Interativo", use_container_width=True, type="secondary")
    with col2:
        gerar_estatico = st.button("Estático", use_container_width=True, type="secondary")

    # -------- MAPA INTERATIVO --------
    if gerar_interativo:
        with st.spinner("Gerando mapa interativo..."):
            m, _ = gerar_mapa_interativo(params, area, geodf)
            if isinstance(m, folium.Map):
                st.session_state["mapa_interativo_html"] = m.get_root().render()
                st.success("✅ Mapa interativo gerado! Veja em 'Resultado atual'.")
                st.rerun()

    # -------- MAPA ESTÁTICO --------
    if gerar_estatico:
        try:
            with st.spinner("Baixando e processando dados..."):
                variable = VAR_MAP_CDS[params["var_label"]]
                nc_bytes = download_era5_data(
                    params['base_dados'], params['modo_agregado'],
                    variable, params['data_params'], params['area']
                )
                ds = abrir_xarray(nc_bytes)

                if params["var_label"] == "Temperatura":
                    da = _coalesce(ds, ["t2m", "2m_temperature"])
                    da = _ensure_time_axis(da)
                    da = _collapse_aux_dims(da)
                    if "time" in da.dims and da.sizes.get("time", 1) > 1:
                        da = da.mean(dim="time", skipna=True)
                    da = da - 273.15
                    da.attrs["units"] = "°C"
                    da.attrs["long_name"] = "Temperatura média 2 m"

                # ===========================
#  PREPARO DA PRECIPITAÇÃO
# ===========================
                elif params["var_label"] == "Precipitação":
                # Seleciona a variável de precipitação (ERA5-Land: "tp" ou "total_precipitation")
                    da = _coalesce(ds, ["tp", "total_precipitation", "total_precipitation_sum"])
                    da = _ensure_time_axis(da)
                    da = _collapse_aux_dims(da)

                # --- Interpretação correta do ERA5-Land ---
                # Para dados horários, "tp" é acumulativo desde 00 UTC.
                # Assim, o total diário é o valor final menos o inicial (m → mm).
                    if "time" in da.dims and da.sizes.get("time", 1) > 1:
                    # Garante que os tempos estejam ordenados
                        da = da.sortby("time")
        
                    # Diferença entre o último e o primeiro passo de tempo
                        da = (da.isel(time=-1) - da.isel(time=0)) * 1000.0  # m → mm
                    else:
                    # Se for produto diário (já agregado)
                        da = da * 1000.0

                    # Metadados
                    da.attrs["units"] = "mm"
                    da.attrs["long_name"] = f"Precipitação acumulada ({params['modo_agregado'].lower()})"


                else:  # Vento
                    u10 = _coalesce(ds, ["u10", "10m_u_component_of_wind"])
                    v10 = _coalesce(ds, ["v10", "10m_v_component_of_wind"])
                    u10 = _ensure_time_axis(u10)
                    v10 = _ensure_time_axis(v10)
                    u10 = _collapse_aux_dims(u10)
                    v10 = _collapse_aux_dims(v10)
                    if "time" in u10.dims and u10.sizes.get("time", 1) > 1:
                        u10 = u10.mean(dim="time", skipna=True)
                        v10 = v10.mean(dim="time", skipna=True)
                    da = np.sqrt(u10**2 + v10**2)
                    da.attrs["units"] = "m/s"
                    da.attrs["long_name"] = "Velocidade do vento (10 m)"

                if da.size == 0:
                    raise ValueError("Nenhum dado retornado.")

                titulo = f"{da.attrs.get('long_name', params['var_label'])} • {params['periodo_str']}"
                fig = plot_da(
                    da, titulo, params["var_label"], geodf=geodf,
                    lat_center=params["area_params"].get("lat_center"),
                    lon_center=params["area_params"].get("lon_center"),
                    raio_km=params["area_params"].get("raio_km")
                )

                # ✅ Salva no estado para aparecer em Resultado atual
                st.session_state["mapa_estatico"] = fig
                st.session_state["mapa_estatico_dados"] = da.to_dataframe(name="valor").reset_index()
                st.success("✅ Mapa estático gerado! Veja em 'Resultado atual'.")
                st.rerun()

        except Exception as e:
            st.error(f"Falha ao gerar mapa estático: {e}")
            st.warning("Verifique se os parâmetros são compatíveis e tente novamente.")

# ==================================================================================
# MAPA INTERATIVO (GEE + FOLIUM)
# ==================================================================================



def gerar_mapa_interativo(params, area, geodf=None):
    var_label = params["var_label"]

    # ===================== Intervalo de datas =====================
    modo_agregado = params["modo_agregado"]
    if modo_agregado == "Mensal":
        y, m = params["data_params"]["year"], params["data_params"]["month"]
        di = date(y, m, 1)
        df_ = date(y, m, pd.Period(f"{y}-{m:02d}").days_in_month)
    elif modo_agregado == "Anual":
        y = params["data_params"]["year"]
        di = date(y, 1, 1)
        df_ = date(y, 12, 31)
    else:  # Personalizado
        di = params["data_params"]["data_inicio"]
        df_ = params["data_params"]["data_fim"]

    # ===================== Seleção da coleção (única) =====================
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(str(di), str(df_))

    # --- Otimização apenas para o modo anual ---
    if modo_agregado == "Anual":
        dias = ee.List.sequence(
            0,
            ee.Date(str(df_)).difference(ee.Date(str(di)), "day")
        ).map(lambda d: ee.Date(str(di)).advance(d, "day"))

        def agrega_diaria(data):
            data = ee.Date(data)
            # Precipitação → soma diária
            soma_tp = collection.filterDate(data, data.advance(1, "day")).select("total_precipitation_hourly").sum()
            # Temperatura → média diária
            media_t2m = collection.filterDate(data, data.advance(1, "day")).select("temperature_2m").mean()
            # Vento → média diária
            u = collection.filterDate(data, data.advance(1, "day")).select("u_component_of_wind_10m").mean()
            v = collection.filterDate(data, data.advance(1, "day")).select("v_component_of_wind_10m").mean()
            diaria = soma_tp.addBands(media_t2m).addBands(u).addBands(v)
            return diaria.set("system:time_start", data.millis())

        collection = ee.ImageCollection(dias.map(agrega_diaria))

    # ===================== Verificação mais segura da coleção =====================
    try:
        tamanho = collection.limit(1).size().getInfo()
        if tamanho == 0:
            st.error("⚠️ Nenhum dado encontrado no GEE para o período selecionado.")
            return None, None
    except Exception as e:
        st.warning(f"⚠️ Não foi possível verificar o tamanho da coleção ({e}). Continuando mesmo assim...")

    # ===================== Seleção da variável =====================
    if var_label == "Temperatura":
        img = collection.select("temperature_2m").mean().subtract(273.15)  # K → °C
        vis = {"min": -20, "max": 55, "palette": ["blue", "white", "red"]}

    elif var_label == "Precipitação":
        img = collection.select("total_precipitation_hourly").sum().multiply(1000)  # m → mm
        vis = {"min": 0, "max": 1000, "palette": ["white", "blue", "darkblue"]}

    elif var_label == "Vento":
        u = collection.select("u_component_of_wind_10m").mean()
        v = collection.select("v_component_of_wind_10m").mean()
        img = u.hypot(v)  # √(u²+v²)
        vis = {"min": 0, "max": 50, "palette": ["white", "yellow", "red"]}

    else:
        st.error("Variável não suportada para mapas interativos.")
        return None, None

    # ===================== Geometria para recorte =====================
    modo_area = params["area_params"]["modo_area"]
    if modo_area == "Estado":
        geom = get_geometry_gee(uf_name=params["area_params"]["selected_uf"])
    elif modo_area == "Município":
        geom = get_geometry_gee(
            uf_name=params["area_params"]["selected_uf"],
            mun_name=params["area_params"]["selected_mun"]
        )
    else:
        N, W, S, E = area
        geom = ee.Geometry.BBox(W, S, E, N)

    img = img.clip(geom)

    # ===================== Estatística real para colorbar/visual =====================
    try:
        band_name = img.bandNames().get(0).getInfo()
        stats = img.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geom,
            scale=9000,
            maxPixels=1e8,
            bestEffort=True
        ).getInfo()

        vmin_real = float(stats.get(f"{band_name}_min"))
        vmax_real = float(stats.get(f"{band_name}_max"))

        if not (vmin_real < vmax_real):
            vmax_real = vmin_real + 1e-6

        vis["min"] = vmin_real
        vis["max"] = vmax_real

    except Exception:
        pass

    # ===================== Criação do mapa Folium =====================
    N, W, S, E = area
    center_lat, center_lon = (N + S) / 2, (W + E) / 2
    try:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            control_scale=True,
            tiles="CartoDB positron",
            attr="© OpenStreetMap © CartoDB"
        )
    except Exception:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # ===================== Camada raster do GEE =====================
    map_id = ee.Image(img).getMapId(vis)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Dados: Google Earth Engine | Fonte: ECMWF ERA5-Land",
        overlay=True,
        name=var_label,
        control=True,
    ).add_to(m)

    # ===================== Contorno (GeoDataFrame) =====================
    try:
        if geodf is not None and not geodf.empty:
            folium.GeoJson(
                data=geodf.__geo_interface__,
                name="Área selecionada",
                style_function=lambda x: {
                    "fillColor": "none",
                    "color": "black",
                    "weight": 2,
                    "dashArray": "5,5"
                },
            ).add_to(m)
            minx, miny, maxx, maxy = geodf.total_bounds
            m.fit_bounds([[miny, minx], [maxy, maxx]])
        else:
            m.fit_bounds([[S, W], [N, E]])
    except Exception as e:
        st.warning(f"Não foi possível aplicar contorno: {e}")
        m.fit_bounds([[S, W], [N, E]])

    folium.LayerControl().add_to(m)

    # ===================== Colorbar (dinâmica com valores reais) =====================
    if var_label == "Precipitação":
        titulo = "Precipitação (mm)"
        paleta_css = "linear-gradient(to right, #f7fbff, #c6dbef, #6baed6, #2171b5, #08306b)"
    elif var_label == "Temperatura":
        titulo = "Temperatura (°C)"
        paleta_css = "linear-gradient(to right, blue, white, red)"
    elif var_label == "Vento":
        titulo = "Velocidade do Vento (m/s)"
        paleta_css = "linear-gradient(to right, white, yellow, red)"
    else:
        titulo = None

    if titulo:
        vmin = float(vis["min"])
        vmax = float(vis["max"])
        t1 = vmin + 0.25 * (vmax - vmin)
        t2 = vmin + 0.50 * (vmax - vmin)
        t3 = vmin + 0.75 * (vmax - vmin)

        colorbar_html = f"""
        <div style="
            position: fixed; 
            bottom: 50px; left: 30px; width: 260px; height: 40px; 
            border: 1px solid #999;
            background: {paleta_css};
            z-index:9999; 
            font-size: 12px;
            text-align: center;
            color: black;">
            <b>{titulo}</b><br>
            {vmin:.1f}&nbsp;&nbsp;&nbsp;&nbsp;
            {t1:.1f}&nbsp;&nbsp;&nbsp;&nbsp;
            {t2:.1f}&nbsp;&nbsp;&nbsp;&nbsp;
            {t3:.1f}&nbsp;&nbsp;&nbsp;&nbsp;
            {vmax:.1f}
        </div>
        """
        m.get_root().html.add_child(folium.Element(colorbar_html))

    return m, img




# ==================================================================================
# TELA: SÉRIES TEMPORAIS
# ==================================================================================


def tela_series():
    st.markdown("---")
    st.markdown(f"## {PAGINAS['series']}")
    st.markdown("Nesta página você pode gerar **séries temporais** de dados climáticos.")

    # ===================== Resultado atual (PERSISTENTE) =====================
    st.markdown("### Resultado atual")
    mostrou_algo = False
    
    if st.button("🗑️ Limpar resultado atual (séries)", type="secondary", key="limpar_series"):
        st.session_state["serie_unica"] = None
        st.session_state["serie_multi"] = None
        st.session_state["series_df"] = None
        st.session_state["series_df_multi"] = None
        st.session_state["series_params"] = None
        st.success("Resultados de séries apagados. Faça uma nova escolha na sidebar.")
        st.rerun()

    # --- Série única (média) ---
    if st.session_state.get("serie_unica") is not None:
        st.pyplot(st.session_state["serie_unica"])
        mostrou_algo = True
        if st.session_state.get("series_df") is not None and st.session_state.get("series_params") is not None:
            params = st.session_state["series_params"]
            df_export = st.session_state["series_df"].copy()
            nome_auto = _clean_filename(
                f"era5_serie_{params['var_label'].lower()}_{params['periodo_str'].replace('/','-')}"
            )
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Série")
            excel_buffer.seek(0)
            export_buttons(nome_auto, csv_data, excel_buffer, plot_fig=st.session_state["serie_unica"])
            with st.expander("Mostrar dados em tabela"):
                st.dataframe(df_export, use_container_width=True)

    # --- Série Mín/Méd/Máx ---
    if st.session_state.get("serie_multi") is not None:
        st.pyplot(st.session_state["serie_multi"])
        mostrou_algo = True
        if st.session_state.get("series_df_multi") is not None and st.session_state.get("series_params") is not None:
            params = st.session_state["series_params"]
            df_export = st.session_state["series_df_multi"].copy()
            nome_auto = _clean_filename(
                f"era5_series_min_med_max_{params['var_label'].lower()}_{params['periodo_str'].replace('/','-')}"
            )
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Série_MMM")
            excel_buffer.seek(0)
            export_buttons(nome_auto, csv_data, excel_buffer, plot_fig=st.session_state["serie_multi"])
            with st.expander("Mostrar dados em tabela (Mín/Méd/Máx)"):
                st.dataframe(df_export, use_container_width=True)

    if not mostrou_algo:
        st.caption("Ainda não há séries geradas nesta sessão.")

    # ===================== Sidebar de parâmetros =====================
    with st.sidebar:
        submitted, params = ui_sidebar_escolhas_series(prefix="series")

    if submitted:
        st.session_state.series_params = params

    # ===================== PRÉ-VISUALIZAÇÃO DA ÁREA =====================
    if not st.session_state.series_params:
        modo_area = params["area_params"]["modo_area"]
        
        # Mostra título e mapa de pré-visualização conforme o modo
        if modo_area == "Círculo (Lat, Lon, Raio)":
            st.subheader("Pré-visualização da Área Circular")
            ui_previews(params, prefix="series")
        elif modo_area == "Polígono Personalizado":
            st.subheader("Desenhe o Polígono de Interesse")
            ui_previews(params, prefix="series")

        st.info("Selecione os parâmetros e clique em **Revisar Parâmetros**.")
        return

    params = st.session_state.series_params
    if not ui_revisao(params):
        return

    # ===================== Determinar área =====================
    area = None
    modo_area = params["area_params"]["modo_area"]
    if modo_area == "Estado":
        collection_states = get_gee_collection_states()
        uf_name = params["area_params"]["selected_uf"]
        area = get_area_from_gee(collection_states, uf_name=uf_name)
    elif modo_area == "Município":
        collection = get_gee_collection()
        uf_name = params["area_params"]["selected_uf"]
        mun_name = params["area_params"]["selected_mun"]
        area = get_area_from_gee(collection, uf_name=uf_name, mun_name=mun_name)
    elif modo_area == "Polígono Personalizado":
        area = st.session_state.area_poligono_series
        if area is None:
            st.warning("Desenhe um polígono e clique em **Revisar Parâmetros**.")
            ui_previews(params, prefix="series")
            return
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        lat_c = params["area_params"]["lat_center"]
        lon_c = params["area_params"]["lon_center"]
        raio_c = params["area_params"]["raio_km"]
        area = get_area_from_radius(lat_c, lon_c, raio_c)

    if area is None:
        st.error("Não foi possível determinar a área. Revise a seleção.")
        return

    params["area"] = area

    # ===================== Botões =====================
    st.warning("⚠️ Dependendo da escolha dos parâmetros, a consulta pode levar alguns minutos.")
    st.subheader("Gerar Séries")
    col1, col2 = st.columns(2)
    with col1:
        gerar_unica = st.button("Média (única)", use_container_width=True, type="secondary")
    with col2:
        gerar_multi = st.button("Mín/Méd/Máx", use_container_width=True, type="secondary")

    # ===================== Geração das Séries =====================
    unidade_map = {"Precipitação": "mm", "Temperatura": "°C", "Vento": "m/s"}
    unidade = unidade_map.get(params["var_label"], "unid.")
    var_cds = VAR_MAP_CDS[params["var_label"]]
    freq = params["data_params"].get("freq_code", "D")  # diário por padrão

    # --- Série única (média espacial diária) ---
    if gerar_unica:
        try:
            with st.spinner("Baixando e processando dados (série única)..."):
                nc_bytes = download_era5_data_diario(
                    params["base_dados"], var_cds, params["data_params"], area
                )
                df = build_series(
                    nc_bytes, params["var_label"], bbox=area, freq=freq, how="Média"
                )
                titulo = f"{params['var_label']} diária • {params['periodo_str']} • {params['area_params']['area_label']}"
                fig = plot_series(df, titulo=titulo, unidade=unidade)
                st.session_state["serie_unica"] = fig
                st.session_state["series_df"] = df.reset_index().rename(columns={"time": "data", "valor": "valor"})
                st.success("✅ Série (média) gerada! Veja em 'Resultado atual'.")
                st.rerun()
        except Exception as e:
            st.error(f"Falha ao gerar série única: {e}")

    # --- Série Mín/Méd/Máx ---
    if gerar_multi:
        try:
            with st.spinner("Baixando e processando dados (Mín/Méd/Máx)..."):
                nc_bytes = download_era5_data_diario(
                    params["base_dados"], var_cds, params["data_params"], area
                )
                df_min = build_series(nc_bytes, params["var_label"], bbox=area, freq=freq, how="Mínimo").rename(columns={"valor": "minimo"})
                df_med = build_series(nc_bytes, params["var_label"], bbox=area, freq=freq, how="Média").rename(columns={"valor": "media"})
                df_max = build_series(nc_bytes, params["var_label"], bbox=area, freq=freq, how="Máximo").rename(columns={"valor": "maximo"})
                df_multi = df_med.join([df_min, df_max], how="outer").sort_index()
                titulo = f"{params['var_label']} diária (Mín/Méd/Máx) • {params['periodo_str']} • {params['area_params']['area_label']}"
                fig = plot_series_multi(df_multi, titulo=titulo, unidade=unidade)
                st.session_state["serie_multi"] = fig
                st.session_state["series_df_multi"] = df_multi.reset_index().rename(columns={"time": "data"})
                st.success("✅ Séries Mín/Méd/Máx geradas! Veja em 'Resultado atual'.")
                st.rerun()
        except Exception as e:
            st.error(f"Falha ao gerar séries Mín/Méd/Máx: {e}")



def tela_sobre():
    st.markdown(f"## {PAGINAS['sobre']}")
    st.markdown(
        """
        ### Clima-Cast-Crepaldi

        Este sistema foi desenvolvido como parte da disciplina  
        **CAT314 – Ferramentas de Previsão de Curtíssimo Prazo (Nowcasting)**  
        
        #### Objetivo do Projeto
        O objetivo é disponibilizar um **dashboard interativo** para monitoramento de variáveis
        meteorológicas (precipitação, temperatura e vento), utilizando dados do 
        **ERA5-Land** processados no **Google Earth Engine (GEE)** e visualizados por meio da
        biblioteca **Streamlit**.  

        #### Funcionalidades
        - Geração de **mapas estáticos** e **mapas interativos** por estado, município ou área personalizada.  
        - Produção de **séries temporais diárias, mensais ou anuais** com estatísticas (mínimo, média, máximo).  
        - **Exportação de dados e figuras** (CSV, XLSX, JPG, PNG, PDF, ZIP).  

        #### Tecnologias Utilizadas
        - [Copernicus Climate Data Store (ERA5-Land)](https://cds.climate.copernicus.eu)  
        - [Google Earth Engine](https://earthengine.google.com/)  
        - [Streamlit](https://streamlit.io/)  
        - [Folium](https://python-visualization.github.io/folium/)  

        ---
        **Autor:** Paulo César Crepaldi  
        **Instituição:** Universidade Federal de Itajubá (UNIFEI)  
        **Disciplina:** CAT314 – Ferramentas de Previsão de Curtíssimo Prazo  
        **Professor:** Dr. Enrique Vieira Mattos  
        """
    )
    st.markdown(
    """
    <div style="
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        font-size: 1.05rem;
        ">
        Esperamos que o <strong>Clima-Cast-Crepaldi</strong> seja uma ferramenta útil para suas análises e estudos meteorológicos!
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================================================
# MAIN
# ==================================================================================

def main():
    # --- Logo centralizado ---
    logo_paths = ["logo.png", "assets/logo.png"]
    logo = next((p for p in logo_paths if os.path.exists(p)), None)

    if logo:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(logo, width=320)
    else:
        st.warning("Logo não encontrado (procurei por logo.png ou assets/logo.png).")

    # --- Menu lateral (só aqui, não repetir depois) ---
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Selecione a página:",
        list(PAGINAS.keys()),
        format_func=lambda x: PAGINAS[x],
        key="menu_navegacao"
    )

    # Inicializa estados
    if "mapas_params" not in st.session_state:
        st.session_state.mapas_params = None
    if "series_params" not in st.session_state:
        st.session_state.series_params = None
    if "area_poligono_mapas" not in st.session_state:
        st.session_state.area_poligono_mapas = None
    if "area_poligono_series" not in st.session_state:
        st.session_state.area_poligono_series = None
    if "mapa_interativo" not in st.session_state:
        st.session_state.mapa_interativo = None
    if "mapa_estatico" not in st.session_state:
        st.session_state.mapa_estatico = None

    # --- Direciona para a página escolhida ---
    if page == "mapas":
        tela_mapas()
    elif page == "series":
        tela_series()
    elif page == "sobre":
        tela_sobre()

# ==================================================================================
# ENTRYPOINT
# ==================================================================================
if __name__ == "__main__":
    main()


