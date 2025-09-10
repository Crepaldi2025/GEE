# app_ccc_mapas_v3b.py
# - Navegação sem callback/rerun
# - Módulo Mapas com preview e export (PNG/JPG/PDF/ZIP)
# - Fallback seguro para map_export_name (sem KeyError)
# - Lembrete: execute com "streamlit run app_ccc_mapas_v3b.py"

import streamlit as st
from datetime import datetime, date
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
import zipfile

# ----------------- CONFIG -----------------
st.set_page_config(page_title="CCC - Clima-Cast-Crepaldi", page_icon="⛈️",
                   layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "CCC - Clima-Cast-Crepaldi"
APP_SUB   = "Monitoramento de temperatura, precipitação e vento"

# ----------------- ESTADO (defaults) -----------------
if "db_base" not in st.session_state: st.session_state.db_base = "ERA5"
if "map_var" not in st.session_state: st.session_state.map_var = "Precipitação"
if "map_agg" not in st.session_state: st.session_state.map_agg = "Acumulado Diário"
if "map_daily_str" not in st.session_state: st.session_state.map_daily_str = date.today().strftime("%Y/%m/%d")
if "map_year" not in st.session_state: st.session_state.map_year = 2025
if "map_month_name" not in st.session_state: st.session_state.map_month_name = "Jan"
if "map_custom_start" not in st.session_state: st.session_state.map_custom_start = date.today().replace(day=1).strftime("%Y/%m/%d")
if "map_custom_end" not in st.session_state: st.session_state.map_custom_end = date.today().strftime("%Y/%m/%d")
if "export_width" not in st.session_state: st.session_state.export_width = 1280

MESES = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
MES2NUM = {m:i+1 for i,m in enumerate(MESES)}
ANOS = list(range(1940, 2026))

# ----------------- STYLE -----------------
st.markdown("""
<style>
section[data-testid="stSidebar"]{background:#fff!important;color:#111!important;min-width:320px;max-width:440px;border-right:1px solid #e5e5e5}
.main .block-container{padding-top:.5rem!important}
.custom-title{font-size:22px;font-weight:700;margin:0}
.custom-sub{font-size:14px;color:#666;margin:2px 0 6px}
.badges{display:flex;gap:.5rem;flex-wrap:wrap}
.badge{display:inline-block;padding:.22rem .55rem;border-radius:999px;font-size:.85rem;border:1px solid #e0e0e0;background:#fafafa}
.badge b{margin-right:.25rem;color:#333}
</style>
""", unsafe_allow_html=True)

# ----------------- Utils -----------------
def _pick_font(sz=22):
    try:
        from PIL import ImageFont
        return ImageFont.truetype("DejaVuSans.ttf", sz)
    except:
        return ImageFont.load_default()

def img_bytes(img, fmt="PNG"):
    bio = BytesIO()
    if fmt=="PNG": img.save(bio, format="PNG")
    elif fmt=="JPG": img.convert("RGB").save(bio, format="JPEG", quality=92)
    elif fmt=="PDF": img.convert("RGB").save(bio, format="PDF")
    return bio.getvalue()

def parse_date_str(s: str):
    from datetime import datetime as dt
    try: return dt.strptime(s.strip(), "%Y/%m/%d").date()
    except: return None

def periodo_label_mapas():
    agg = st.session_state.map_agg
    if agg=="Acumulado Diário":
        dt_ = parse_date_str(st.session_state.map_daily_str)
        return dt_.strftime("%Y/%m/%d") if dt_ else "Data inválida"
    if agg=="Acumulado Mensal":
        return f"{st.session_state.map_year}-{MES2NUM.get(st.session_state.map_month_name,1):02d}"
    if agg=="Acumulado Anual":
        return f"{st.session_state.map_year}"
    d1=parse_date_str(st.session_state.map_custom_start); d2=parse_date_str(st.session_state.map_custom_end)
    s1=d1.strftime("%Y/%m/%d") if d1 else "inválida"; s2=d2.strftime("%Y/%m/%d") if d2 else "inválida"
    return f"{s1} — {s2}"

def render_placeholder_map(base, var, agg, periodo, width_px=1280, ratio=16/9):
    w = int(width_px); h = int(width_px/ratio)
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    t1=_pick_font(36); t2=_pick_font(22); t3=_pick_font(18)
    d.text((40, 40), f"Mapa — {var} • {agg}", fill=(0,0,0), font=t1)
    d.text((40, 96), f"Base: {base}   |   Período: {periodo}", fill=(10,10,10), font=t2)
    d.text((40,130), f"Gerado em {datetime.now(ZoneInfo('America/Sao_Paulo')):%d/%m/%Y %H:%M}", fill=(90,90,90), font=t3)
    d.rectangle([30,170,w-30,h-30], outline=(180,180,180), width=2)
    d.text((50,185), "Preview (placeholder — camada real aqui)", fill=(120,120,120), font=t2)
    return im

def nome_auto_mapa():
    base, var, agg = st.session_state.db_base, st.session_state.map_var, st.session_state.map_agg
    per = periodo_label_mapas()
    s = f"mapa_{base}_{var}_{agg}_{per}".lower().replace(" ","_").replace("/","")
    return s

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("#### Página Inicial")
    st.caption(f"**Data/Hora:** {datetime.now(ZoneInfo('America/Sao_Paulo')):%d/%m/%Y %H:%M}")
    st.divider()

    nav_choice = st.radio("Escolha a seção:", ["🌍 Mapas Interativos", "📊 Séries Temporais"], index=0)
    # — vamos focar Mapas; Séries ficaram ocultas neste exemplo

    if nav_choice.startswith("🌍"):
        st.markdown("### Menu (Mapas)")
        st.selectbox("Base de Dados:", ["ERA5"], key="db_base")
        st.selectbox("Variável:", ["Precipitação","Temperatura","Vento"], key="map_var")
        st.selectbox("Tipo de acumulado:", ["Acumulado Diário","Acumulado Mensal","Acumulado Anual","Acumulado Personalizado"], key="map_agg")

        st.divider(); st.markdown("### Período")
        if st.session_state.map_agg=="Acumulado Diário":
            st.text_input("Data (YYYY/MM/DD):", key="map_daily_str", placeholder="YYYY/MM/DD")
            if parse_date_str(st.session_state.map_daily_str) is None:
                st.warning("Use YYYY/MM/DD (ex.: 2024/09/10).")
        elif st.session_state.map_agg=="Acumulado Mensal":
            st.select_slider("Ano:", options=ANOS, key="map_year")
            st.radio("Mês:", MESES, key="map_month_name", horizontal=True)
        elif st.session_state.map_agg=="Acumulado Anual":
            st.select_slider("Ano:", options=ANOS, key="map_year")
        else:
            st.text_input("Data inicial (YYYY/MM/DD):", key="map_custom_start", placeholder="YYYY/MM/DD")
            st.text_input("Data final (YYYY/MM/DD):",   key="map_custom_end",  placeholder="YYYY/MM/DD")
            d1=parse_date_str(st.session_state.map_custom_start); d2=parse_date_str(st.session_state.map_custom_end)
            if d1 is None or d2 is None: st.warning("Use YYYY/MM/DD nas duas datas.")
            elif d1>d2: st.warning("Data inicial deve ser ≤ data final.")

        st.divider(); st.markdown("### Exportar")
        # Fallback seguro: se a chave ainda não existir, usa o nome automático
        default_name = nome_auto_mapa()
        name_current = st.session_state.get("map_export_name", default_name)
        st.text_input("Nome do arquivo:", value=name_current, key="map_export_name")

        st.slider("Largura da imagem (px):", 960, 2560, value=st.session_state.export_width, step=160, key="export_width")

# ----------------- HEADER -----------------
st.markdown(f"<div class='custom-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='custom-sub'>{APP_SUB}</div>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- CORPO (Mapas) -----------------
if nav_choice.startswith("🌍"):
    periodo = periodo_label_mapas()
    st.markdown("#### Resumo (Mapas)")
    st.markdown(
        f"""<div class="badges">
            <span class="badge"><b>Base:</b> {st.session_state.db_base}</span>
            <span class="badge"><b>Variável:</b> {st.session_state.map_var}</span>
            <span class="badge"><b>Acumulado:</b> {st.session_state.map_agg}</span>
            <span class="badge"><b>Período:</b> {periodo}</span>
        </div>""", unsafe_allow_html=True)

    # Preview placeholder
    img_map = render_placeholder_map(
        st.session_state.db_base,
        st.session_state.map_var,
        st.session_state.map_agg,
        periodo,
        width_px=st.session_state.export_width
    )
    st.image(img_map, caption="Preview do mapa (placeholder)", use_container_width=True)

    # Arquivos p/ export
    png = img_bytes(img_map,"PNG")
    jpg = img_bytes(img_map,"JPG")
    pdf = img_bytes(img_map,"PDF")

    # Use SEMPRE um fallback local para o nome (evita KeyError)
    map_name = st.session_state.get("map_export_name", nome_auto_mapa())

    zipb = BytesIO()
    with zipfile.ZipFile(zipb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{map_name}.png", png)
        zf.writestr(f"{map_name}.jpg", jpg)
        zf.writestr(f"{map_name}.pdf", pdf)

    fmt = st.selectbox("Formato:", ["PNG","JPG","PDF","ZIP (PNG+JPG+PDF)"], key="map_fmt")
    if fmt=="PNG":
        st.download_button("⬇️ Baixar PNG", data=png,
                           file_name=f"{map_name}.png",
                           mime="image/png", use_container_width=True)
    elif fmt=="JPG":
        st.download_button("⬇️ Baixar JPG", data=jpg,
                           file_name=f"{map_name}.jpg",
                           mime="image/jpeg", use_container_width=True)
    elif fmt=="PDF":
        st.download_button("⬇️ Baixar PDF", data=pdf,
                           file_name=f"{map_name}.pdf",
                           mime="application/pdf", use_container_width=True)
    else:
        st.download_button("⬇️ Baixar ZIP", data=zipb.getvalue(),
                           file_name=f"{map_name}.zip",
                           mime="application/zip", use_container_width=True)

else:
    st.info("A aba de Séries foi omitida nesta versão de teste. Troque para **Mapas Interativos** na barra lateral.")
