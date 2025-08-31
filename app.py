# app.py — GEE + Streamlit (Service Account) + Mapa SRTM
# Requer: streamlit, earthengine-api, geemap, folium

import json
import streamlit as st
import ee

st.set_page_config(page_title="GEE + Streamlit — SRTM", layout="wide")
st.title("GEE + Streamlit — SRTM")
st.caption(f"Projeto: {st.secrets.get('EARTHENGINE_PROJECT', 'gee-crepaldi-2025')}")

# ----------------- AUTENTICAÇÃO (Service Account via secrets) -----------------
def init_ee() -> None:
    service_account = st.secrets["EE_SERVICE_ACCOUNT"]
    private_key_json = st.secrets["EE_PRIVATE_KEY"]   # JSON inteiro entre """..."""
    project_id = st.secrets.get("EARTHENGINE_PROJECT", "gee-crepaldi-2025")

    # pode vir como string → carregar para dict
    key_dict = json.loads(private_key_json)

    credentials = ee.ServiceAccountCredentials(service_account, key_dict)
    ee.Initialize(credentials, project=project_id)
    st.success("Earth Engine inicializado via Service Account.")

try:
    init_ee()
except Exception as e:
    st.error(
        "Falha ao inicializar o Earth Engine com Service Account.\n"
        "Confira os *Secrets* e os papéis IAM (Service Usage Consumer e, para dev, Editor)."
    )
    st.exception(e)
    st.stop()

# ----------------- APP (Mapa + consulta) -----------------
import geemap.foliumap as geemap

with st.sidebar:
    st.header("Navegar")
    lat = st.number_input("Latitude", value=-22.90, format="%.6f")
    lon = st.number_input("Longitude", value=-46.30, format="%.6f")
    zoom = st.slider("Zoom", 3, 12, 8)
    if st.button("Ir para LAT/LON"):
        st.session_state._go_to = (lon, lat, zoom)

    st.divider()
    st.subheader("Consulta de elevação")
    lat_q = st.number_input("Lat (consulta)", value=-22.75, format="%.6f", key="latq")
    lon_q = st.number_input("Lon (consulta)", value=-46.32, format="%.6f", key="lonq")
    if st.button("Consultar SRTM"):
        pt = ee.Geometry.Point([lon_q, lat_q])
        srtm = ee.Image("CGIAR/SRTM90_V4")
        try:
            elev = (srtm.sample(pt, 90).first().get("elevation").getInfo())
            st.info(f"Elevação aproximada: **{elev:.1f} m**")
        except Exception as e:
            st.error("Não foi possível obter a elevação.")
            st.exception(e)

# Cria mapa folium
m = geemap.Map(plugin_Draw=True, locate_control=True, draw_export=True)

# Camada SRTM
srtm = ee.Image("CGIAR/SRTM90_V4")
m.addLayer(srtm, {"min": 0, "max": 3000}, "Elevação SRTM")
m.centerObject(srtm, 3)

# Centralizar se o usuário pediu
if "_go_to" in st.session_state:
    lon_g, lat_g, zoom_g = st.session_state._go_to
    m.set_center(lon_g, lat_g, zoom_g)

st.write("Mapa interativo:")
m.to_streamlit(height=650)

st.caption("Se algo falhar, abra **⋮ → Manage app → Logs** para detalhes.")
