# app.py — GEE + Streamlit (Service Account) + Mapa SRTM
import json
import tempfile

import ee
import streamlit as st
import geemap.foliumap as geemap


# ----------------- CONFIGURAÇÃO DO APP -----------------
st.set_page_config(page_title="GEE + Streamlit — SRTM", layout="wide")
st.title("GEE + Streamlit — SRTM")
st.caption(f"Projeto: {st.secrets['earthengine']['project_id']}")


# ----------------- AUTENTICAÇÃO (Service Account via secrets) -----------------
def init_ee():
    
    try:
        # Lê o bloco inteiro dos secrets
        sa_info = dict(st.secrets["earthengine"])

        # Cria um JSON temporário exatamente no formato que a API espera
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(sa_info, f)
            f.flush()
            credentials = ee.ServiceAccountCredentials(sa_info["client_email"], f.name)
            ee.Initialize(credentials, project=sa_info["project_id"])

        st.success("Earth Engine inicializado via Service Account.")
        return True

    except Exception as e:
        st.error("Falha ao inicializar o Earth Engine com Service Account.")
        st.exception(e)
        return False


if not init_ee():
    st.stop()


# ----------------- APP (Mapa + consulta) -----------------
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

# Mapa interativo
m = geemap.Map(plugin_Draw=True, locate_control=True, draw_export=True)

# Camada SRTM
srtm = ee.Image("CGIAR/SRTM90_V4")
m.addLayer(srtm, {"min": 0, "max": 3000}, "Elevação SRTM")
m.centerObject(srtm, 3)

# Centralizar se o usuário clicou
if "_go_to" in st.session_state:
    lon_g, lat_g, zoom_g = st.session_state._go_to
    m.set_center(lon_g, lat_g, zoom_g)

st.write("Mapa interativo:")
m.to_streamlit(height=650)

st.caption("Se algo falhar, verifique **⋮ → Manage app → Logs**.")
