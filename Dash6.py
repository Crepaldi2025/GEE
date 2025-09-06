# app_ccc_fluxo_series_e_mapas_export_apenas.py
# Séries Temporais (Variável → Local → Estatísticas) e
# Mapas Interativos (Tipo → Produto → Período → Exportar)
# Sem consultas externas; apenas navegação e exportações (placeholders).

import streamlit as st
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os

# =============================
# CONFIG GERAL
# =============================
st.set_page_config(
    page_title="CCC - Clima-Cast-Crepaldi",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "CCC - Clima-Cast-Crepaldi"
APP_SUB   = "Monitoramento de temperatura, precipitação e vento"

# Caminhos de imagens (ajuste conforme seu ambiente)
PATH_LOGO  = r"C:\Users\crepa\Desktop\git\CAT314\Logo.jpg"
PATH_SERIE = r"C:\Users\crepa\Desktop\git\CAT314\Serie.png"
PATH_MAPA  = r"C:\Users\crepa\Desktop\git\CAT314\Mapa.png"

# Ajuste do topo (rem; negativos reduzem espaço)
MARGIN_TOP_REM = -10

# =============================
# ESTADOS
# =============================
if "pagina" not in st.session_state:
    st.session_state.pagina = "inicio"   # "inicio" | "series" | "mapas"

# Séries Temporais
if "series_step" not in st.session_state:
    st.session_state.series_step = "variavel"   # "variavel" | "regiao" | "estatisticas"
if "series_var" not in st.session_state:
    st.session_state.series_var = "Precipitação"
if "series_regiao" not in st.session_state:
    st.session_state.series_regiao = None
if "estat_view" not in st.session_state:
    st.session_state.estat_view = None          # "Médias" | "Médias Móveis" | "Tendências"

# Mapas Interativos (novo fluxo: tipo → produto → período → exportar)
if "map_step" not in st.session_state:
    st.session_state.map_step = "tipo"          # "tipo" | "produto" | "periodo" | "export"
if "map_tipo" not in st.session_state:
    st.session_state.map_tipo = "Satélite"
if "map_produto" not in st.session_state:
    st.session_state.map_produto = "GOES-19 (VIS)"      # restrito conforme pedido
if "map_periodo" not in st.session_state:
    hoje = date.today()
    st.session_state.map_periodo = (hoje - timedelta(days=3), hoje)  # (ini, fim)

def ir_para(pagina: str):
    st.session_state.pagina = pagina
    # reset ao sair
    if pagina != "series":
        st.session_state.series_step = "variavel"
        st.session_state.series_var = "Precipitação"
        st.session_state.series_regiao = None
        st.session_state.estat_view = None
    if pagina != "mapas":
        st.session_state.map_step = "tipo"
        st.session_state.map_tipo = "Satélite"
        st.session_state.map_produto = "GOES-19 (VIS)"
        hoje = date.today()
        st.session_state.map_periodo = (hoje - timedelta(days=3), hoje)

def series_step_ir(step: str):
    st.session_state.series_step = step
    if step != "estatisticas":
        st.session_state.estat_view = None

def map_step_ir(step: str):
    st.session_state.map_step = step

# =============================
# CSS GLOBAL
# =============================
st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{ min-width: 420px; max-width: 420px; }}
section[data-testid="stSidebar"] > div:first-child {{ height: 100vh; overflow-y: hidden; }}
section[data-testid="stSidebar"] ::-webkit-scrollbar {{ width: 0px; height: 0px; }}

.main .block-container {{
    padding-top: 0rem !important;
    margin-top: {MARGIN_TOP_REM}rem !important;
}}

.custom-title {{ font-size:22px; font-weight:700; margin:0; }}
.custom-sub   {{ font-size:14px; color:gray; margin: 2px 0 4px 0; }}
hr {{ margin-top: 6px !important; margin-bottom: 6px !important; }}
.custom-hr {{ margin-top:6px; margin-bottom:8px; border:none; border-top:1px solid #bbb; }}
h1, h2, h3, h4 {{ margin-top: 6px; margin-bottom: 6px; }}
</style>
""", unsafe_allow_html=True)

# =============================
# Utils
# =============================
def safe_load_image(path: str, width: int = 120):
    try:
        if not (path and os.path.exists(path)):
            return None
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        scale = width / float(w)
        new_h = max(1, int(h * scale))
        return img.resize((width, new_h), Image.LANCZOS)
    except Exception:
        return None

def png_bytes_from_image(img: Image.Image) -> bytes:
    bio = BytesIO(); img.save(bio, format="PNG"); return bio.getvalue()

def pdf_bytes_from_image(img: Image.Image) -> bytes:
    bio = BytesIO(); img.convert("RGB").save(bio, format="PDF"); return bio.getvalue()

# Placeholders (sem dados)
def imagem_placeholder_serie(variavel: str, estat: str, regiao: dict) -> Image.Image:
    w, h = 1200, 675
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    try:
        font_big = ImageFont.truetype("arial.ttf", 38)
        font_md  = ImageFont.truetype("arial.ttf", 22)
        font_sm  = ImageFont.truetype("arial.ttf", 18)
    except:
        font_big = ImageFont.load_default(); font_md = ImageFont.load_default(); font_sm = ImageFont.load_default()
    d.text((40, 40), f"{variavel} — {estat}", fill=(0,0,0), font=font_big)
    d.text((40, 96), "Região: (não aplicado aqui)", fill=(0,0,0), font=font_md)
    d.text((40, 136), f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", fill=(90,90,90), font=font_sm)
    d.rectangle([30, 30, w-30, h-30], outline=(200,200,200), width=2)
    d.rectangle([60, 180, w-60, h-60], outline=(150,150,150), width=2)
    d.text((70, 190), "Gráfico placeholder (sem dados).", fill=(120,120,120), font=font_md)
    return im

def imagem_placeholder_mapa_sem_area(tipo: str, produto: str, periodo_txt: str) -> Image.Image:
    """Placeholder simples (sem AOI)."""
    w, h = 1280, 720
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    try:
        font_big = ImageFont.truetype("arial.ttf", 36)
        font_md  = ImageFont.truetype("arial.ttf", 22)
    except:
        font_big = ImageFont.load_default(); font_md = ImageFont.load_default()
    d.text((40, 40), f"Mapa — {tipo} / {produto}", fill=(0,0,0), font=font_big)
    d.text((40, 90), f"Período: {periodo_txt}", fill=(0,0,0), font=font_md)
    d.rectangle([30, 150, w-30, h-30], outline=(180,180,180), width=2)
    d.text((50, 165), "Preview (placeholder — sem camada real e sem AOI)", fill=(120,120,120), font=font_md)
    return im

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("### Navegação")
    now = datetime.now(ZoneInfo("America/Sao_Paulo"))
    st.caption(f"**Data/Hora:** {now:%d/%m/%Y %H:%M}")
    st.divider()

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        if (im := safe_load_image(PATH_SERIE, 90)): st.image(im)
    with c2:
        if st.button("Séries temporais", use_container_width=True):
            ir_para("series"); st.rerun()

    st.write("")
    c3, c4 = st.columns([1.2, 1.8])
    with c3:
        if (im := safe_load_image(PATH_MAPA, 90)): st.image(im)
    with c4:
        if st.button("Mapas interativos", use_container_width=True):
            ir_para("mapas"); st.rerun()

# =============================
# Cabeçalho com Logo
# =============================
col_logo, col_txt = st.columns([0.18, 1])
with col_logo:
    if (lg := safe_load_image(PATH_LOGO, 70)): st.image(lg)
with col_txt:
    st.markdown(f"<div class='custom-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='custom-sub'>{APP_SUB}</div>", unsafe_allow_html=True)
st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

# =============================
# Widget de Região (para Séries; Mapas não precisa mais)
# =============================
UF_LIST = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
           "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"]

def widget_regiao(key_prefix=""):
    modo = st.radio(
        "Modo de seleção de região:",
        ["Estado", "Município", "LAT/LON (ponto + raio)", "Polígono (GeoJSON)"],
        horizontal=True,
        key=f"{key_prefix}reg_modo"
    )
    reg = {"modo": modo}
    if modo == "Estado":
        uf = st.selectbox("UF", UF_LIST, index=UF_LIST.index("SP"), key=f"{key_prefix}reg_uf")
        reg.update({"uf": uf})
    elif modo == "Município":
        c1, c2 = st.columns([1, 2])
        with c1:
            uf = st.selectbox("UF", UF_LIST, index=UF_LIST.index("SP"), key=f"{key_prefix}reg_mun_uf")
        with c2:
            municipios_mock = ["(selecione)", "São Paulo", "Campinas", "Santos"]
            muni = st.selectbox("Município", municipios_mock, key=f"{key_prefix}reg_mun_nome")
        reg.update({"uf": uf, "municipio": muni})
    elif modo == "LAT/LON (ponto + raio)":
        c1, c2, c3 = st.columns(3)
        with c1:
            lat = st.number_input("Latitude", -90.0, 90.0, -22.42, step=0.01, key=f"{key_prefix}reg_lat")
        with c2:
            lon = st.number_input("Longitude", -180.0, 180.0, -45.45, step=0.01, key=f"{key_prefix}reg_lon")
        with c3:
            raio_km = st.number_input("Raio (km)", 0.1, 500.0, 10.0, step=0.5, key=f"{key_prefix}reg_raio")
        reg.update({"lat": lat, "lon": lon, "raio_km": raio_km})
    else:
        pass
    return reg

# =============================
# PÁGINA: Séries Temporais
# =============================
def page_series():
    st.subheader("Séries temporais")
    st.divider()

    step = st.session_state.series_step

    # 1) VARIÁVEL
    if step == "variavel":
        var = st.radio(
            "Escolha a variável:",
            ["Precipitação", "Temperatura", "Vento"],
            index=["Precipitação","Temperatura","Vento"].index(st.session_state.series_var),
            horizontal=True,
            key="var_escolha"
        )
        col_a, col_b = st.columns([1.2, 1.8])
        with col_a:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()
        with col_b:
            if st.button("➡️ Confirmar variável e escolher local"):
                st.session_state.series_var = var
                series_step_ir("regiao"); st.rerun()

    # 2) REGIÃO
    elif step == "regiao":
        reg = widget_regiao(key_prefix="series_")
        st.divider()
        col1, col2, col3 = st.columns([1.1, 1.1, 1.8])
        with col1:
            if st.button("⬅️ Voltar (Variável)"):
                series_step_ir("variavel"); st.rerun()
        with col2:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()
        with col3:
            if st.button("✅ Confirmar local e ir para Estatísticas"):
                st.session_state.series_regiao = reg
                series_step_ir("estatisticas"); st.rerun()

    # 3) ESTATÍSTICAS + EXPORTAR
    elif step == "estatisticas":
        reg = st.session_state.series_regiao or {}
        var = st.session_state.series_var

        # Cabeçalho curto
        if reg.get("modo") == "Estado":
            st.markdown(f"**Região:** UF **{reg.get('uf')}**")
        elif reg.get("modo") == "Município":
            st.markdown(f"**Região:** **{reg.get('municipio')}** – **{reg.get('uf')}**")
        elif reg.get("modo") == "LAT/LON (ponto + raio)":
            st.markdown(f"**Região:** Lat/Lon **{reg.get('lat'):.4f}**, **{reg.get('lon'):.4f}** • Raio **{reg.get('raio_km')} km**")
        else:
            st.markdown("**Região:** Polígono (GeoJSON)")
        st.markdown(f"**Variável:** {var}")

        # Estatísticas (apenas seleção)
        st.markdown("### Estatísticas")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📊 Médias"):
                st.session_state.estat_view = "Médias"; st.rerun()
        with c2:
            if st.button("📈 Médias Móveis"):
                st.session_state.estat_view = "Médias Móveis"; st.rerun()
        with c3:
            if st.button("📉 Tendências"):
                st.session_state.estat_view = "Tendências"; st.rerun()

        # Exportar
        if st.session_state.estat_view:
            st.markdown(f"**Estatística selecionada:** {st.session_state.estat_view}")

            csv_bytes = ( "# CSV modelo (sem dados)\n"
                          "date,value\n"
                          "2020-01-01,\n2020-01-02,\n2020-01-03,\n" ).encode("utf-8")
            st.download_button(
                "⬇️ Exportar CSV (modelo)",
                data=csv_bytes,
                file_name=f"serie_{var.lower()}_{st.session_state.estat_view.lower().replace(' ','_')}.csv",
                mime="text/csv",
                key="series_csv"
            )

            img = imagem_placeholder_serie(var, st.session_state.estat_view, reg)
            st.download_button(
                "🖼️ Exportar PNG (gráfico placeholder)",
                data=png_bytes_from_image(img),
                file_name=f"grafico_{var.lower()}_{st.session_state.estat_view.lower().replace(' ','_')}.png",
                mime="image/png",
                key="series_png"
            )
            st.download_button(
                "📄 Exportar PDF",
                data=pdf_bytes_from_image(img),
                file_name=f"relatorio_{var.lower()}_{st.session_state.estat_view.lower().replace(' ','_')}.pdf",
                mime="application/pdf",
                key="series_pdf"
            )

        st.divider()
        c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
        with c1:
            if st.button("⬅️ Voltar (Local)"):
                series_step_ir("regiao"); st.rerun()
        with c2:
            if st.button("↩️ Voltar (Variável)"):
                series_step_ir("variavel"); st.rerun()
        with c3:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()

# =============================
# PÁGINA: Mapas Interativos (Tipo → Produto → Período → Exportar)
# =============================
def page_mapas():
    st.subheader("Mapas interativos")
    st.divider()

    step = st.session_state.map_step

    # 1) Tipo
    if step == "tipo":
        tipo = st.radio("Tipo de mapa:", ["Satélite", "Radar"], horizontal=True, key="map_tipo_radio",
                        index=["Satélite","Radar"].index(st.session_state.map_tipo))
        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()
        with c2:
            if st.button("➡️ Confirmar tipo"):
                st.session_state.map_tipo = tipo
                map_step_ir("produto"); st.rerun()

    # 2) Produto (restrito: GOES-19 VIS/IR/WV ou Radar CAPPI 3 km)
    elif step == "produto":
        if st.session_state.map_tipo == "Satélite":
            produto = st.selectbox(
                "Produto (Satélite — GOES-19)",
                ["GOES-19 (VIS)", "GOES-19 (IR)", "GOES-19 (WV)"],
                index=["GOES-19 (VIS)", "GOES-19 (IR)", "GOES-19 (WV)"].index(st.session_state.map_produto),
                key="map_prod_sel"
            )
        else:
            produto = st.selectbox(
                "Produto (Radar)",
                ["Refletividade CAPPI 3 km"],
                index=0, key="map_prod_sel"
            )
        c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
        with c1:
            if st.button("⬅️ Voltar (Tipo)"):
                map_step_ir("tipo"); st.rerun()
        with c2:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()
        with c3:
            if st.button("➡️ Confirmar produto"):
                st.session_state.map_produto = produto
                map_step_ir("periodo"); st.rerun()

    # 3) Período (datas) → vai direto para EXPORTAR
    elif step == "periodo":
        ini_default, fim_default = st.session_state.map_periodo
        periodo = st.date_input(
            "Período (data inicial e final):",
            value=(ini_default, fim_default),
            key="map_periodo_input"
        )
        if isinstance(periodo, tuple) and len(periodo) == 2:
            data_ini, data_fim = periodo
        else:
            data_ini, data_fim = ini_default, fim_default

        c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
        with c1:
            if st.button("⬅️ Voltar (Produto)"):
                map_step_ir("produto"); st.rerun()
        with c2:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()
        with c3:
            if st.button("➡️ Confirmar período"):
                st.session_state.map_periodo = (data_ini, data_fim)
                map_step_ir("export"); st.rerun()

    # 4) EXPORTAR (sem área/visualização)
    elif step == "export":
        tipo    = st.session_state.map_tipo
        produto = st.session_state.map_produto
        data_ini, data_fim = st.session_state.map_periodo
        periodo_txt = f"{data_ini.strftime('%d/%m/%Y')} — {data_fim.strftime('%d/%m/%Y')}"

        st.markdown(f"**Tipo:** {tipo}  •  **Produto:** {produto}  •  **Período:** {periodo_txt}")
        st.markdown("### Exportar")

        # Placeholder simples para gerar PNG/PDF
        img = imagem_placeholder_mapa_sem_area(tipo, produto, periodo_txt)

        base_name = (produto.replace(" ", "_")
                            .replace("(", "")
                            .replace(")", "")
                            .lower())

        st.download_button(
            "🖼️ PNG",
            data=png_bytes_from_image(img),
            file_name=f"mapa_{tipo.lower()}_{base_name}_{data_ini:%Y%m%d}-{data_fim:%Y%m%d}.png",
            mime="image/png",
            key="map_png"
        )
        st.download_button(
            "📄 PDF",
            data=pdf_bytes_from_image(img),
            file_name=f"mapa_{tipo.lower()}_{base_name}_{data_ini:%Y%m%d}-{data_fim:%Y%m%d}.pdf",
            mime="application/pdf",
            key="map_pdf"
        )

        st.divider()
        c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
        with c1:
            if st.button("⬅️ Voltar (Período)"):
                map_step_ir("periodo"); st.rerun()
        with c2:
            if st.button("↩️ Voltar (Produto)"):
                map_step_ir("produto"); st.rerun()
        with c3:
            if st.button("🏠 Início"):
                ir_para("inicio"); st.rerun()

# =============================
# Roteamento
# =============================
if st.session_state.pagina == "series":
    page_series()
elif st.session_state.pagina == "mapas":
    page_mapas()
else:
    st.subheader("Início")
    st.write("Selecione uma opção na barra lateral para começar.")
