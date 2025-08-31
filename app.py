# app.py — Streamlit + Earth Engine + geemap
import streamlit as st
import ee

st.set_page_config(page_title="GEE + Streamlit", layout="wide")
st.title("GEE + Streamlit — SRTM")
st.caption("Projeto: gee-crepaldi-2025")

# ---------- Inicialização ----------
# Tenta Service Account (deploy). Se não houver secrets, cai no OAuth (local).
def init_ee():
    try:
        service_account = st.secrets["EE_SERVICE_ACCOUNT"]
        private_key = st.secrets["EE_PRIVATE_KEY"]
        project_id = st.secrets.get("EARTHENGINE_PROJECT", "gee-crepaldi-2025")
        credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
        ee.Initialize(credentials, project=project_id)
        st.success("Earth Engine inicializado (Service Account).")
    except Exception:
        st.info("Usando OAuth local. Se solicitado, autorize no navegador.")
        try:
            ee.Initialize(project="gee-crepaldi-2025")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project="gee-crepaldi-2025")
        st.success("Earth Engine inicializado (OAuth local).")

init_ee()

# ---------- Mapa ----------
import geemap.foliumap as geemap

m = geemap.Map(plugin_Draw=True, locate_control=True, draw_export=True)
srtm = ee.Image("CGIAR/SRTM90_V4")
m.centerObject(srtm, 3)
m.addLayer(srtm, {"min": 0, "max": 3000}, "Elevação SRTM")

with st.sidebar:
    st.header("Localizar ponto")
    lat = st.number_input("Latitude", value=-22.90, format="%.6f")
    lon = st.number_input("Longitude", value=-46.30, format="%.6f")
    if st.button("Ir para LAT/LON"):
        m.set_center(lon, lat, 9)

st.write("Mapa (SRTM 90 m)")
m.to_streamlit(height=650)
