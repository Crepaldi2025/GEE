# app.py — GEE + Streamlit (Service Account) + Mapa SRTM

import streamlit as st
import ee

st.set_page_config(page_title="GEE + Streamlit — SRTM", layout="wide")
st.title("GEE + Streamlit — SRTM")
st.caption(f"Projeto: {st.secrets.get('EARTHENGINE_PROJECT', 'gee-crepaldi-2025')}")

# ----------------- AUTENTICAÇÃO (Service Account via secrets) -----------------
def init_ee() -> bool:
    try:
        credentials = ee.ServiceAccountCredentials(
            st.secrets["EE_SERVICE_ACCOUNT"],       # e-mail da SA
            key_data=st.secrets["EE_PRIVATE_KEY"]   # JSON completo como STRING
        )
        ee.Initialize(credentials, project=st.secrets["EARTHENGINE_PROJECT"])
        st.success("Earth Engine inicializado via Service Account.")
        return True
    except Exception as e:
        st.error(
            "Falha ao inicializar o Earth Engine com Service Account.\n"
            "Confira os *Secrets* (EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY, EARTHENGINE_PROJECT) "
            "e os papéis IAM (Service Usage Consumer e, para dev, Editor)."
        )
        st.exception(e)
        return False

if not init_ee():
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
    st.subheader("Consulta de elevação (SRTM)")
    lat_q = st.number_input("Lat (consulta)", value=-22.75, format="%.6f", key="latq")
    lon_q = st.number_input("Lon (consulta)", value=-46.32, format="%.6f", key="lonq")
    if st.button("Consultar"):
        pt = ee.Geometry.Point([lon_q, lat_q])
        srtm_img = ee.Image("CGIAR/SRTM90_V4")
        try:
            elev = srtm_img.sample(pt, 90).first().get("elevation").getInfo()
            st.info(f"Elevação aproximada: **{elev:.1f} m**")
        except Exception as e:
            st.error("Não foi possível obter a elevação.")
            st.exception(e)

# Mapa folium via geemap
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

st.caption("Se algo falhar, verifique **⋮ → Manage app → Logs**.")
