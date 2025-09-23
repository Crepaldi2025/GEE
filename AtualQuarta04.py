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


# ===================== INICIALIZAÇÕES E CONFIGURAÇÃO INICIAL =====================
try:
    ee.Initialize(project='gee-crepaldi-2025')
except Exception as e:
    st.error(f"Erro ao inicializar o Earth Engine: {e}. Verifique a instalação/autenticação.")
    st.stop()

st.set_page_config(page_title="CCC - Clima-Cast-Crepaldi", page_icon="⛈️", layout="wide")
APP_TITLE = "CCC - Clima-Cast-Crepaldi"
PAGINAS = {"home": "Início", "mapas": "Mapas Interativos", "series": "Séries Temporais"}

# ===================== GERENCIAMENTO DE ESTADO DA SESSÃO =====================
defaults = {
    "page": "home",
    "area_poligono_mapas": None,
    "area_poligono_series": None,
    "mapas_params": None,
    "series_params": None,
    "series_df": None,          # DataFrame de série única
    "series_df_multi": None,    # DataFrame com colunas Min/Med/Max
    "series_how": "Média",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== CONSTANTES E METADADOS =====================
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
MES2NUM = {m: i + 1 for i, m in enumerate(MESES)}
ANOS = list(range(1940, datetime.now().year + 1))

# nomes para pedir ao CDS (download)
VAR_MAP_CDS = {
    "Temperatura": "2m_temperature",
    "Precipitação": "total_precipitation",
    "Vento": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
}

# ===================== UTILS GERAIS =====================
def _clean_filename(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("—", "-").replace("–", "-").replace("/", "")
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    return s

def _normalize_string(s: str) -> str:
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')

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
        try: os.remove(tmp_path)
        except: pass

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
    north = lat + delta_lat; south = lat - delta_lat
    east = lon + delta_lon;  west  = lon - delta_lon
    return [north, west, south, east]  # ordem CDS

# ===================== DOWNLOAD DE DADOS (CDS) =====================
@st.cache_data(show_spinner="Baixando dados do ERA5...", max_entries=10)
def download_era5_data(base_dados, modo_agregado, variable, data_params, area):
    """
    Estratégia: sempre usar 'reanalysis-era5-single-levels' (horário) e agregar localmente.
    Evita ambiguidades do produto 'monthly means' para precipitação.
    """
    c = cds_client()
    dataset = "reanalysis-era5-single-levels"
    req = {
        "format": "netcdf",
        "product_type": "reanalysis",
        "grid": [0.25, 0.25],
        "area": area,
    }

    req["variable"] = tuple(variable) if isinstance(variable, list) else variable

    # janela de datas
    if modo_agregado == "Diário":
        y = data_params["year"]; m = data_params["month"]; d = data_params["day"]
        di = date(y, m, d); df_ = di
    elif modo_agregado == "Mensal":
        y = data_params["year"]; m = data_params["month"]
        di = date(y, m, 1)
        df_ = date(y, m, pd.Period(f"{y}-{m:02d}").days_in_month)
    elif modo_agregado == "Anual":
        y = data_params["year"]; di = date(y, 1, 1); df_ = date(y, 12, 31)
    else:  # Personalizado
        di = data_params["data_inicio"]; df_ = data_params["data_fim"]

    if di > df_:
        raise ValueError("A data de início não pode ser posterior à data de fim.")

    req["date"] = f"{di:%Y-%m-%d}/{df_:%Y-%m-%d}"
    req["time"] = [f"{h:02d}:00" for h in range(24)]

    result = c.retrieve(dataset, req)
    return _result_to_bytes(result)

def abrir_xarray(nc_bytes: bytes) -> xr.Dataset:
    return xr.open_dataset(BytesIO(nc_bytes))

# ===================== HELPERS DE VARIÁVEL/TEMPO =====================
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
    """Remove/colapsa dimensões e coords auxiliares (expver/number)."""
    if "expver" in da.dims:
        da = da.max("expver", skipna=True)
    if "number" in da.dims:
        da = da.mean("number", skipna=True)
    # remove coords homônimas para não virarem colunas no to_dataframe
    drop_coords = [c for c in ("expver", "number") if c in da.coords and c not in da.dims]
    if drop_coords:
        da = da.reset_coords(drop=True)
    return da

# ===================== PRÉ-PROCESSAMENTO PARA MAPAS =====================
def preparar_da_mapa(ds: xr.Dataset, var_label: str, modo_agregado: str, data_params: dict) -> xr.DataArray:
    # Variável e unidade
    if var_label == "Temperatura":
        da = _coalesce(ds, ["t2m", "2m_temperature"]) - 273.15
        unidade = "°C"; lname = "Temperatura média 2 m"
    elif var_label == "Precipitação":
        da = _coalesce(ds, ["tp", "total_precipitation"])  # m
        unidade = "mm"; lname = f"Precipitação acumulada ({modo_agregado.lower()})"
    else:
        u10 = _coalesce(ds, ["u10", "10m_u_component_of_wind"])
        v10 = _coalesce(ds, ["v10", "10m_v_component_of_wind"])
        da = np.sqrt(u10**2 + v10**2)
        unidade = "m/s"; lname = "Velocidade do vento (média 10 m)"

    # Normaliza tempo e dims auxiliares
    da = _ensure_time_axis(da)
    da = _collapse_aux_dims(da)

    # Agrega no tempo
    if "time" in da.dims and da.sizes.get("time", 1) > 1:
        if var_label == "Precipitação":
            da = da.sum(dim="time", skipna=True) * 1000.0
        else:
            da = da.mean(dim="time", skipna=True)
    elif var_label == "Precipitação":
        da = da * 1000.0

    # Padroniza nomes lat/lon
    lat_name = "latitude" if "latitude" in da.dims else ("lat" if "lat" in da.dims else None)
    lon_name = "longitude" if "longitude" in da.dims else ("lon" if "lon" in da.dims else None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Dimensões de latitude/longitude não encontradas. Dims: {da.dims}")
    if lat_name != "lat": da = da.rename({lat_name: "lat"})
    if lon_name != "lon": da = da.rename({lon_name: "lon"})

    da = da.squeeze()
    da.attrs["units"] = unidade
    da.attrs["long_name"] = lname
    return da

# ===================== CÁLCULO DE SÉRIES TEMPORAIS =====================
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
        da_mm = da * 1000.0  # m → mm
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
    else:  # Precipitação
        da = _coalesce(ds, ["tp", "total_precipitation"])

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

# ===================== PLOTAGEM E EXPORTAÇÃO =====================
def plot_da(da: xr.DataArray, titulo: str, var_label: str, geodf=None):
    if da.size == 0:
        raise ValueError("Mapa vazio: nenhum dado retornado para a seleção.")

    fig = plt.figure(figsize=(10, 8))
    lon_min, lon_max = float(da['lon'].min()), float(da['lon'].max())
    lat_min, lat_max = float(da['lat'].min()), float(da['lat'].max())

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    try:
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='darkgray', linestyle='--')
    except Exception:
        pass

    lat_values = da['lat'].values
    lon_values = da['lon'].values
    data_values = da.values
    if lat_values[0] > lat_values[-1]:
        lat_values = lat_values[::-1]; data_values = data_values[::-1, :]

    if data_values.size == 0:
        raise ValueError("Sem dados após recorte espacial.")

    vmin = float(np.nanmin(data_values)); vmax = float(np.nanmax(data_values))
    if np.isclose(vmin, vmax): vmin, vmax = None, None

    cmap_map = {"Temperatura": "coolwarm", "Precipitação": "YlGnBu", "Vento": "YlOrRd"}
    im = ax.imshow(
        data_values,
        transform=ccrs.PlateCarree(),
        extent=[lon_values.min(), lon_values.max(), lat_values.min(), lat_values.max()],
        cmap=cmap_map.get(var_label, "viridis"), origin='lower', vmin=vmin, vmax=vmax
    )

    if geodf is not None:
        ax.add_geometries(geodf.geometry, crs=ccrs.PlateCarree(),
                          edgecolor='dimgray', facecolor='none', linewidth=1.2)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xformatter = LongitudeFormatter(); gl.yformatter = LatitudeFormatter()

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.05)
    cb.set_label(da.attrs.get("units", "Unidade"))

    ax.set_title(titulo)
    fig.tight_layout()
    return fig

def plot_series(df: pd.DataFrame, titulo: str, unidade: str, xlim=None):
    fig, ax = plt.subplots(figsize=(9,4))
    n = len(df.index)
    use_marker = n <= 2
    ax.plot(df.index, df["valor"], marker='o' if use_marker else None)
    if xlim is not None:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    elif n == 1:
        t = pd.to_datetime(df.index[0]); pad = pd.Timedelta(days=15)
        ax.set_xlim(t - pad, t + pad)
    ax.set_title(titulo)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(unidade)
    ax.grid(True, alpha=.3, linestyle="--")
    fig.tight_layout()
    return fig

def plot_series_multi(df_multi: pd.DataFrame, titulo: str, unidade: str, xlim=None):
    fig, ax = plt.subplots(figsize=(10,4.5))
    n = len(df_multi.index)
    use_marker = n <= 2
    if "minimo" in df_multi: ax.plot(df_multi.index, df_multi["minimo"], label="Mínimo", marker='o' if use_marker else None)
    if "media"  in df_multi: ax.plot(df_multi.index, df_multi["media"],  label="Média",  marker='o' if use_marker else None)
    if "maximo" in df_multi: ax.plot(df_multi.index, df_multi["maximo"], label="Máximo", marker='o' if use_marker else None)
    if xlim is not None:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    elif n == 1:
        t = pd.to_datetime(df_multi.index[0]); pad = pd.Timedelta(days=15)
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

# ===================== INTEGRAÇÃO COM GOOGLE EARTH ENGINE =====================
@st.cache_data(show_spinner=False)
def get_gee_collection():
    return ee.FeatureCollection("FAO/GAUL/2015/level2")

_UF_PT_TO_GEE = {
    'Amapá': 'Amapa','Ceará': 'Ceara','Espírito Santo': 'Espirito Santo','Goiás': 'Goias',
    'Maranhão': 'Maranhao','Paraíba': 'Paraiba','Paraná': 'Parana','Piauí': 'Piaui',
    'Rondônia': 'Rondonia','São Paulo': 'Sao Paulo','Pará': 'Para','Rio Grande do Sul': 'Rio Grande do Sul',
    'Minas Gerais': 'Minas Gerais','Mato Grosso do Sul': 'Mato Grosso do Sul','Mato Grosso': 'Mato Grosso',
    'Rio Grande do Norte': 'Rio Grande do Norte','Rio de Janeiro': 'Rio de Janeiro','Santa Catarina': 'Santa Catarina',
    'Bahia': 'Bahia','Alagoas': 'Alagoas','Sergipe': 'Sergipe','Pernambuco': 'Pernambuco','Tocantins': 'Tocantins',
    'Acre': 'Acre','Amazonas': 'Amazonas','Roraima': 'Roraima','Distrito Federal': 'Distrito Federal'
}
_GEE_TO_PT_FIX = {
    'Amapa': 'Amapá','Ceara': 'Ceará','Espirito Santo': 'Espírito Santo','Goias': 'Goiás',
    'Maranhao': 'Maranhão','Paraiba': 'Paraíba','Parana': 'Paraná','Piaui': 'Piauí',
    'Rondonia': 'Rondônia','Sao Paulo': 'São Paulo','Para': 'Pará'
}

@st.cache_data(show_spinner=False)
def get_uf_names(_collection):
    brazil_regions = _collection.filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
    uf_list_gee = brazil_regions.aggregate_array('ADM1_NAME').distinct().sort().getInfo()
    return sorted([_GEE_TO_PT_FIX.get(name, name) for name in uf_list_gee])

@st.cache_data(show_spinner=False)
def get_state_polygon_gee(_collection, uf_name):
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    ee_feature = _collection.filter(ee.Filter.eq('ADM1_NAME', gee_uf_name))
    json_geom = ee_feature.geometry().getInfo()
    return gpd.GeoDataFrame.from_features([{'geometry': json_geom, 'properties': {}}], crs="EPSG:4326")

@st.cache_data(show_spinner=False)
def get_mun_names(_collection, uf_name):
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    mun_list = _collection.filter(
        ee.Filter.And(ee.Filter.eq('ADM0_NAME', 'Brazil'), ee.Filter.eq('ADM1_NAME', gee_uf_name))
    ).aggregate_array('ADM2_NAME').distinct().sort().getInfo()
    return sorted(mun_list)

def get_area_from_gee(_collection, uf_name=None, mun_name=None):
    gee_uf_name = _UF_PT_TO_GEE.get(uf_name, uf_name)
    gee_mun_name = _normalize_string(mun_name) if mun_name else None

    if uf_name and not mun_name:
        feature = _collection.filter(ee.Filter.eq('ADM1_NAME', gee_uf_name)).geometry()
    elif uf_name and mun_name:
        feature = _collection.filter(
            ee.Filter.And(ee.Filter.eq('ADM1_NAME', gee_uf_name), ee.Filter.eq('ADM2_NAME', gee_mun_name))
        ).geometry()
    else:
        feature = _collection.filter(ee.Filter.eq('ADM0_NAME', 'Brazil')).geometry()

    bounds = feature.bounds().getInfo()['coordinates'][0]
    lat_max = bounds[2][1]; lon_min = bounds[0][0]
    lat_min = bounds[0][1]; lon_max = bounds[2][0]
    return [lat_max, lon_min, lat_min, lon_max]

# ===================== UI COMPARTILHADA (SIDEBAR) =====================
def ui_sidebar_escolhas(prefix: str):
    st.header("Menu (Escolhas)")
    st.caption(f"Agora: {datetime.now(ZoneInfo('America/Sao_Paulo')):%d/%m/%Y %H:%M}")

    st.markdown("### Base de Dados")
    base_dados = st.selectbox("Base de Dados", ["ERA5"], key=f"{prefix}_base", label_visibility="hidden")

    st.markdown("### Variável")
    var_label = st.selectbox("Variável", ["Precipitação", "Temperatura", "Vento"], key=f"{prefix}_var", label_visibility="hidden")

    st.markdown("### Agregado temporal")
    modo_agregado = st.selectbox("Agregado temporal", ["Diário", "Mensal", "Anual", "Personalizado"], key=f"{prefix}_agregado", label_visibility="hidden")

    st.markdown("### Período")
    periodo_str = ""
    data_params = {}
    freq_code = None  # para "Personalizado"

    if modo_agregado == "Diário":
        dt = st.date_input("Data (YYYY/MM/DD):", value=date.today(), key=f"{prefix}_data_d")
        periodo_str = dt.strftime('%Y/%m/%d')
        data_params = {"year": dt.year, "month": dt.month, "day": dt.day}
    elif modo_agregado == "Mensal":
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
        di = st.date_input("Data de Início (YYYY/MM/DD):", value=date(2021,1,1), key=f"{prefix}_inicio_p")
        df_ = st.date_input("Data de Fim (YYYY/MM/DD):", value=date(2021,1,31), key=f"{prefix}_fim_p")
        periodo_str = f"de {di.strftime('%d/%m/%Y')} a {df_.strftime('%d/%m/%Y')}"
        # seletor de frequência
        freq_label = st.selectbox(
            "Frequência da série (Personalizado)",
            ["Horária", "Diária", "Mensal"],
            index=1, key=f"{prefix}_freq_p"
        )
        freq_map = {"Horária": "H", "Diária": "D", "Mensal": "MS"}
        freq_code = freq_map[freq_label]
        data_params = {"data_inicio": di, "data_fim": df_, "freq_code": freq_code}

    st.markdown("### Área de Interesse")
    modo_area = st.radio("Selecione o modo de seleção:",
                         ["Brasil","Estado","Município","Polígono Personalizado","Círculo (Lat, Lon, Raio)"],
                         key=f"{prefix}_modo_area")
    
    area_params = {"modo_area": modo_area}
    collection = get_gee_collection()
    
    if modo_area == "Brasil":
        area_params["area_label"] = "Brasil"
    elif modo_area == "Estado":
        uf_list = get_uf_names(collection)
        selected_uf = st.selectbox("Selecione o Estado:", uf_list, key=f"{prefix}_estado")
        area_params["selected_uf"] = selected_uf
        area_params["area_label"] = f"Estado de {selected_uf}"
    elif modo_area == "Município":
        uf_list = get_uf_names(collection)
        selected_uf = st.selectbox("Selecione o Estado:", uf_list, key=f"{prefix}_estado2")
        area_params["selected_uf"] = selected_uf
        if selected_uf:
            mun_list = get_mun_names(collection, selected_uf)
            selected_mun = st.selectbox("Selecione o Município:", mun_list, key=f"{prefix}_mun")
            area_params["selected_mun"] = selected_mun
            area_params["area_label"] = f"Município de {selected_mun}/{selected_uf}"
    elif modo_area == "Polígono Personalizado":
        area_params["area_label"] = "Polígono Personalizado"
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        lat_center = st.number_input("Latitude:", value=-22.0, format="%.4f", key=f"{prefix}_lat")
        lon_center = st.number_input("Longitude:", value=-46.0, format="%.4f", key=f"{prefix}_lon")
        raio_km = st.number_input("Raio (km):", value=100.0, min_value=1.0, format="%.2f", key=f"{prefix}_raio")
        area_params.update({"lat_center": lat_center, "lon_center": lon_center, "raio_km": raio_km})
        area_params["area_label"] = f"Círculo (Lat: {lat_center}, Lon: {lon_center}, Raio: {raio_km} km)"

    with st.form(f"{prefix}_form"):
        st.markdown("---")
        submitted = st.form_submit_button("Revisar Parâmetros", use_container_width=True, type="primary")

    params = {
        "base_dados": base_dados,
        "var_label": var_label,
        "modo_agregado": modo_agregado,
        "periodo_str": periodo_str,
        "data_params": data_params,
        "area_params": area_params
    }
    # inclui freq_code somente se for Personalizado
    if freq_code is not None:
        params["data_params"]["freq_code"] = freq_code
    return submitted, params

def ui_previews(params: dict, prefix: str):
    modo_area = params['area_params']['modo_area']
    if modo_area == "Polígono Personalizado":
        st.subheader("Desenhe seu Polígono no Mapa")
        st.info("Use as ferramentas de desenho. Clique em 'Revisar Parâmetros' para confirmar.")
        m = folium.Map(location=[-14.235, -51.9253], zoom_start=4, control_scale=True)
        draw = folium.plugins.Draw(
            position='topleft',
            draw_options={'polyline': False, 'rectangle': False, 'circle': False,
                          'circlemarker': False, 'marker': False,
                          'polygon': {'showArea': True}},
            edit_options={'edit': False, 'remove': True}
        )
        m.add_child(draw)
        out = st_folium(m, width=800, height=520, returned_objects=['all_drawings'])
        if out and out.get("all_drawings"):
            poly_data = out["all_drawings"][0]
            if poly_data['properties']['subType'] == 'polygon':
                coords = poly_data['geometry']['coordinates'][0]
                lats = [c[1] for c in coords]; lons = [c[0] for c in coords]
                st.session_state[f"area_poligono_{prefix}"] = [max(lats), min(lons), min(lats), max(lons)]
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        st.subheader("Pré-visualização da Área Circular")
        lat0 = params['area_params'].get("lat_center", -14.235)
        lon0 = params['area_params'].get("lon_center", -51.9253)
        rkm  = params['area_params'].get("raio_km", 100.0)
        m = folium.Map(location=[lat0, lon0], zoom_start=6, control_scale=True)
        folium.Marker([lat0, lon0], tooltip="Centro").add_to(m)
        pts = _circle_polyline(lat0, lon0, rkm, n=240)
        folium.PolyLine(pts, color="#000000", weight=2, opacity=0.9, dash_array="8,6").add_to(m)
        north, west, south, east = get_area_from_radius(lat0, lon0, rkm)
        m.fit_bounds([[south, west], [north, east]])
        st_folium(m, width=800, height=520)

def ui_revisao(params: dict) -> bool:
    st.markdown("#### Revisão dos Parâmetros")
    if params["modo_agregado"] == "Personalizado":
        di = params["data_params"]["data_inicio"]
        df_ = params["data_params"]["data_fim"]
        if di > df_:
            st.error("Erro: A data de início não pode ser posterior à data de fim.")
            return False
            
    st.markdown(
        f"""
        <div style="display:flex;gap:.5rem;flex-wrap:wrap">
          <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Base: {params['base_dados']}</span>
          <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Variável: {params['var_label']}</span>
          <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Agregado: {params['modo_agregado']}</span>
          <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Período: {params['periodo_str']}</span>
          <span style="padding:.2rem .6rem;border:1px solid #ddd;border-radius:999px">Área: {params['area_params']['area_label']}</span>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    return True

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
            with c1: st.download_button("⬇️ PNG", data=png, file_name=f"{file_name}.png", mime="image/png", use_container_width=True)
            with c2: st.download_button("⬇️ JPG", data=jpg, file_name=f"{file_name}.jpg", mime="image/jpeg", use_container_width=True)
            with c3: st.download_button("⬇️ PDF", data=pdf, file_name=f"{file_name}.pdf", mime="application/pdf", use_container_width=True)
            with c4: st.download_button("⬇️ ZIP", data=zipb.getvalue(), file_name=f"{file_name}.zip", mime="application/zip", use_container_width=True)
        
        with col2:
            st.markdown("#### Dados")
            c1d, c2d = st.columns(2)
            with c1d: st.download_button("⬇️ CSV", data=csv_data, file_name=f"{file_name}.csv", mime="text/csv", use_container_width=True)
            with c2d: st.download_button("⬇️ XLSX", data=excel_buffer, file_name=f"{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    else:
        st.markdown("---")
        c1d, c2d = st.columns(2)
        with c1d: st.download_button("⬇️ CSV", data=csv_data, file_name=f"{file_name}.csv", mime="text/csv", use_container_width=True)
        with c2d: st.download_button("⬇️ XLSX", data=excel_buffer, file_name=f"{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ===================== TELA: MAPAS =====================
def tela_mapas():
    st.markdown("---")
    st.markdown(f"## {PAGINAS['mapas']}")
    st.markdown("Nesta página você pode gerar **mapas** com dados climáticos.")

    with st.sidebar:
        submitted, params = ui_sidebar_escolhas(prefix="mapas")

    if submitted:
        st.session_state.mapas_params = params
    
    if not st.session_state.mapas_params:
        ui_previews(params, prefix="mapas")
        st.info("Selecione os parâmetros na barra lateral e clique em **Revisar Parâmetros**.")
        return

    params = st.session_state.mapas_params
    if not ui_revisao(params):
        return
    
    collection = get_gee_collection()
    area = None
    geodf = None
    modo_area = params["area_params"]["modo_area"]
    
    if modo_area == "Brasil":
        area = get_area_from_gee(collection)
    elif modo_area == "Estado":
        uf_name = params["area_params"]["selected_uf"]
        area = get_area_from_gee(collection, uf_name=uf_name)
        geodf = get_state_polygon_gee(collection, uf_name)
    elif modo_area == "Município":
        uf_name = params["area_params"]["selected_uf"]
        mun_name = params["area_params"]["selected_mun"]
        area = get_area_from_gee(collection, uf_name=uf_name, mun_name=mun_name)
    elif modo_area == "Polígono Personalizado":
        area = st.session_state.area_poligono_mapas
        if area is None:
            st.warning("Desenhe um polígono e clique em **Revisar Parâmetros**.")
            ui_previews(params, prefix="mapas")
            return
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        area = get_area_from_radius(params["area_params"]["lat_center"], params["area_params"]["lon_center"], params["area_params"]["raio_km"])

    if area is None:
        st.error("Não foi possível determinar a área. Por favor, revise a seleção.")
        return

    params['area'] = area

    if st.button("Gerar Mapa", use_container_width=True, type="primary"):
        try:
            with st.spinner("Baixando e processando dados..."):
                variable = VAR_MAP_CDS[params["var_label"]]
                nc_bytes = download_era5_data(params['base_dados'], params['modo_agregado'], variable, params['data_params'], params['area'])
                ds = abrir_xarray(nc_bytes)
                da = preparar_da_mapa(ds, params["var_label"], params["modo_agregado"], params['data_params'])

                if da.size == 0:
                    raise ValueError("Nenhum dado retornado para a seleção.")

            titulo = f"{da.attrs.get('long_name', params['var_label'])} • {params['periodo_str']}"
            fig = plot_da(da, titulo, params["var_label"], geodf=geodf)
            st.pyplot(fig, use_container_width=True)
            
            df_export = da.to_dataframe(name="valor").reset_index()
            nome_auto = _clean_filename(f"era5_{params['var_label'].lower()}_{params['modo_agregado'].replace(' ', '_').lower()}_{params['periodo_str'].replace('/', '-')}")
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Dados')
            excel_buffer.seek(0)
            
            export_buttons(nome_auto, csv_data, excel_buffer, plot_fig=fig)

            with st.expander("Mostrar dados em tabela"):
                st.dataframe(df_export, use_container_width=True)
        
        except Exception as e:
            st.error(f"Falha ao gerar mapa: {e}")
            st.warning("Verifique se os parâmetros (variável, período e área) são compatíveis e tente novamente.")

# ===================== TELA: SÉRIES TEMPORAIS =====================
def tela_series_temporais():
    st.markdown("---")
    st.markdown(f"## {PAGINAS['series']}")
    st.markdown("Gere **séries temporais** agregadas por área.")

    # ====== CSS para colorir os botões ======
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) button {
            background-color: #1f77b4 !important; color: #ffffff !important; border-color: #1f77b4 !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) button {
            background-color: #e0e0e0 !important; color: #111111 !important; border-color: #bdbdbd !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(3) button {
            background-color: #d62728 !important; color: #ffffff !important; border-color: #d62728 !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4) button {
            background-color: #7e57c2 !important; color: #ffffff !important; border-color: #7e57c2 !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"] button:hover { filter: brightness(0.95); }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        submitted, params = ui_sidebar_escolhas(prefix="series")

    if submitted:
        st.session_state.series_params = params
    
    if not st.session_state.series_params:
        ui_previews(params, prefix="series")
        st.info("Selecione os parâmetros e clique em **Revisar Parâmetros**.")
        return
    
    params = st.session_state.series_params
    if not ui_revisao(params):
        return

    # BBox
    collection = get_gee_collection()
    area = None
    modo_area = params["area_params"]["modo_area"]
    
    if modo_area == "Brasil":
        area = get_area_from_gee(collection)
    elif modo_area == "Estado":
        area = get_area_from_gee(collection, uf_name=params["area_params"]["selected_uf"])
    elif modo_area == "Município":
        area = get_area_from_gee(collection, uf_name=params["area_params"]["selected_uf"], mun_name=params["area_params"]["selected_mun"])
    elif modo_area == "Polígono Personalizado":
        area = st.session_state.area_poligono_series
        if area is None:
            st.warning("Desenhe um polígono e clique em **Revisar Parâmetros**.")
            ui_previews(params, prefix="series")
            return
    elif modo_area == "Círculo (Lat, Lon, Raio)":
        area = get_area_from_radius(params["area_params"]["lat_center"], params["area_params"]["lon_center"], params["area_params"]["raio_km"])

    if area is None:
        st.error("Não foi possível determinar a área. Por favor, revise a seleção.")
        return

    params['area'] = area

    # Botões de agregação espacial (agora 5 colunas)
    c1, c2, c3, c4, c5 = st.columns(5)
    gen_min  = c1.button("Mínimo", use_container_width=True)
    gen_med  = c2.button("Média", use_container_width=True)
    gen_max  = c3.button("Máximo", use_container_width=True)
    gen_all  = c4.button("Gerar MIN/MÉD/MÁX", use_container_width=True)
    limpar   = c5.button("Limpar Série", use_container_width=True)
    
    if limpar:
        st.session_state.series_df = None
        st.session_state.series_df_multi = None
        st.session_state.series_how = "Média"
        st.rerun()
    
    how = None
    if gen_med: how = "Média"
    elif gen_min: how = "Mínimo"
    elif gen_max: how = "Máximo"

    # Frequência conforme pedido
    freq_map_fixed = {
        "Diário": "H",   # série horária no dia selecionado
        "Mensal": "D",   # série diária no mês selecionado
        "Anual":  "MS",  # série mensal no ano selecionado
    }
    if params["modo_agregado"] in freq_map_fixed:
        freq = freq_map_fixed[params["modo_agregado"]]
    else:
        # Personalizado com seletor
        freq = params["data_params"].get("freq_code", "D")

    # Determina período de download
    if params["modo_agregado"] == "Diário":
        y, m, d = params["data_params"]["year"], params["data_params"]["month"], params["data_params"]["day"]
        di, df_ = date(y, m, d), date(y, m, d)
        xlim_plot = (pd.Timestamp(di), pd.Timestamp(di) + pd.Timedelta(hours=23))
    elif params["modo_agregado"] == "Mensal":
        y, m = params["data_params"]["year"], params["data_params"]["month"]
        di = date(y, m, 1)
        df_ = date(y, m, pd.Period(f"{y}-{m:02d}").days_in_month)
        xlim_plot = (pd.Timestamp(di), pd.Timestamp(df_))
    elif params["modo_agregado"] == "Anual":
        y = params["data_params"]["year"]; di, df_ = date(y,1,1), date(y,12,31)
        # janela mensal: do primeiro ao último mês
        xlim_plot = (pd.Timestamp(di), pd.Timestamp(df_))
    else:
        di, df_ = params["data_params"]["data_inicio"], params["data_params"]["data_fim"]
        xlim_plot = None  # deixa autoscale

    if di > df_:
        st.error("A data de início não pode ser posterior à data de fim.")
        return

    # Acionadores
    if how is not None:
        try:
            variable = VAR_MAP_CDS[params["var_label"]]
            with st.spinner("Baixando e processando dados para a série..."):
                nc_bytes = download_era5_data(params['base_dados'], "Personalizado", variable, {"data_inicio": di, "data_fim": df_}, params['area'])
                df = build_series(nc_bytes, params["var_label"], params['area'], freq=freq, how=how)
            st.session_state.series_df = df
            st.session_state.series_df_multi = None
            st.session_state.series_how = how
        except Exception as e:
            st.error(f"Falha ao gerar série: {e}")
            st.warning("Verifique parâmetros (variável, período e área) e tente novamente.")
            return

    if gen_all:
        try:
            variable = VAR_MAP_CDS[params["var_label"]]
            with st.spinner("Baixando e processando dados (três séries)..."):
                nc_bytes = download_era5_data(params['base_dados'], "Personalizado", variable, {"data_inicio": di, "data_fim": df_}, params['area'])
                df_min = build_series(nc_bytes, params["var_label"], params['area'], freq=freq, how="Mínimo").rename(columns={"valor":"minimo"})
                df_med = build_series(nc_bytes, params["var_label"], params['area'], freq=freq, how="Média").rename(columns={"valor":"media"})
                df_max = build_series(nc_bytes, params["var_label"], params['area'], freq=freq, how="Máximo").rename(columns={"valor":"maximo"})
                df_multi = df_min.join(df_med, how="outer").join(df_max, how="outer").sort_index()
            st.session_state.series_df = None
            st.session_state.series_df_multi = df_multi
        except Exception as e:
            st.error(f"Falha ao gerar séries Mín/Méd/Máx: {e}")
            st.warning("Verifique parâmetros (variável, período e área) e tente novamente.")
            return

    # Títulos amigáveis
    modo_txt = params["modo_agregado"]
    if modo_txt == "Diário":  modo_txt = "Horária (dia selecionado)"
    if modo_txt == "Mensal":  modo_txt = "Diária (mês selecionado)"
    if modo_txt == "Anual":   modo_txt = "Mensal (ano selecionado)"
    unidade_map = {"Precipitação": "mm", "Temperatura": "°C", "Vento": "m/s"}

    # Renderizações
    if st.session_state.series_df is not None:
        df = st.session_state.series_df
        how_label = st.session_state.get("series_how", "Média")
        unidade = unidade_map[params["var_label"]]
        titulo = f"{params['var_label']} • {how_label} espacial • {modo_txt}"
        fig = plot_series(df, titulo, unidade, xlim=xlim_plot)
        st.pyplot(fig, use_container_width=True)

        nome_auto = _clean_filename(f"serie_{params['var_label'].lower()}_{how_label.lower()}_{params['modo_agregado'].lower()}_{df.index.min():%Y%m%d}_{df.index.max():%Y%m%d}")
        csv_bytes = df.reset_index().rename(columns={"time":"data"}).to_csv(index=False).encode("utf-8")
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.reset_index().rename(columns={"time":"data"}).to_excel(writer, index=False, sheet_name='Série')
        excel_buffer.seek(0)
        
        export_buttons(nome_auto, csv_bytes, excel_buffer)

        with st.expander("Mostrar dados em tabela"):
            st.dataframe(df, use_container_width=True)

    if st.session_state.series_df_multi is not None:
        dfm = st.session_state.series_df_multi.copy()
        unidade = unidade_map[params["var_label"]]
        titulo = f"{params['var_label']} • Mín/Méd/Máx • {modo_txt}"
        figm = plot_series_multi(dfm, titulo, unidade, xlim=xlim_plot)
        st.pyplot(figm, use_container_width=True)

        nome_auto = _clean_filename(f"serie_{params['var_label'].lower()}_min_med_max_{params['modo_agregado'].lower()}_{dfm.index.min():%Y%m%d}_{dfm.index.max():%Y%m%d}")
        csv_bytes = dfm.reset_index().rename(columns={"time":"data"}).to_csv(index=False).encode("utf-8")
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            dfm.reset_index().rename(columns={"time":"data"}).to_excel(writer, index=False, sheet_name='Séries')
        excel_buffer.seek(0)
        
        export_buttons(nome_auto, csv_bytes, excel_buffer)

        with st.expander("Mostrar dados em tabela"):
            st.dataframe(dfm, use_container_width=True)

def tela_home():
    logo_url = "https://raw.githubusercontent.com/Crepaldi2025/GEE/main/assets/Logo.jpg"
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            st.image(logo_url, width=150)
        except Exception:
            st.markdown("### ⛅ CCC")
    with col2:
        st.markdown(f"# {APP_TITLE}")
        st.caption("Bem-vindo ao portal de visualização de dados climáticos")
    st.markdown("---")
    st.markdown("Use o menu para acessar **Mapas Interativos** e **Séries Temporais**.")

# ===================== LÓGICA PRINCIPAL DO APLICATIVO =====================
with st.sidebar:
    st.header("Navegação")
    st.session_state.page = st.radio("Selecione a tela:", PAGINAS.keys(), format_func=lambda x: PAGINAS[x])

if st.session_state.page == "home":
    tela_home()
elif st.session_state.page == "mapas":
    tela_mapas()
elif st.session_state.page == "series":
    tela_series_temporais()
