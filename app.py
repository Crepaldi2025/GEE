# app.py — GEE + Streamlit (Service Account robusto) + Mapa SRTM
# Requer: streamlit, earthengine-api, geemap, folium

import json
import streamlit as st
import ee

st.set_page_config(page_title="GEE + Streamlit — SRTM", layout="wide")
st.title("GEE + Streamlit — SRTM")
st.caption(f"Projeto: {st.secrets.get('EARTHENGINE_PROJECT', 'gee-crepaldi-2025')}")

# ----------------------------------------------------------
#  AUTENTICAÇÃO ROBUSTA COM SERVICE ACCOUNT
#  Aceita EE_PRIVATE_KEY como:
#   - string JSON (TOML """...""" ou '''...''')
#   - objeto já parseado (alguns Streamlit parseiam secrets como dict)
#  Normaliza o campo "private_key" para conter quebras de linha reais.
# ----------------------------------------------------------
def init_ee() -> bool:
    try:
        service_account = st.secrets["EE_SERVICE_ACCOUNT"]
        project_id = st.secrets["EARTHENGINE_PROJECT"]
        key_blob = st.secrets["EE_PRIVATE_KEY"]

        # 1) Garantir que temos um dict com o JSON
        if isinstance(key_blob, str):
            # Remove BOM/acidental spaces
            key_blob = key_blob.strip()
            # Se vier string JSON, parseia
            if key_blob.startswith("{") and key_blob.endswith("}"):
                info = json.loads(key_blob)
            else:
                # Caso extremo: user colou só o PEM (não é o nosso caso)
                raise ValueError("EE_PRIVATE_KEY não parece um JSON da conta de serviço.")
        elif isinstance(key_blob, (dict,)):
            info = dict(key_blob)  # cópia
        else:
            raise ValueError("EE_PRIVATE_KEY precisa ser string JSON ou dict.")

        # 2) Normalizar o campo private_key
        pk = info.get("private_key", "")
        if not pk:
            raise ValueError("Campo 'private_key' ausente no JSON.")

        # Se veio com '\n' literais, converte para quebras reais
        # (ex.: '-----BEGIN...\\n...\\n-----END-----\\n' -> com quebras)
        if "\\n" in pk and "\n" not in pk:
            pk = pk.replace("\\n", "\n")
        # Garante que começa e termina com os marcadores PEM
        if "BEGIN PRIVATE KEY" not in pk:
            raise ValueError("Conteúdo de 'private_key' não parece um PEM válido.")

        info["private_key"] = pk  # salva de volta

        # 3) Inicializar o EE com o JSON normalizado
        credentials = ee.ServiceAccountCredentials(service_account, key_data=json.dumps(info))
        ee.Initialize(credentials, project=project_id)
        st.success("Earth Engine inicializado via Service Account.")
        return True

    except Exception as e:
        st.error(
            "Falha ao inicializar o Earth Engine com Service Account.\n"
            "Revise os *Secrets* e papéis IAM (Service Usage Consumer e, p/ dev, Editor)."
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
