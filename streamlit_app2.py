"""
Application Streamlit — Comparaison de segmentation U-Net vs SegFormer.

Permet d'uploader une photo et affiche côte à côte :
- l'image originale
- la prédiction du modèle U-Net (overlay + masque)
- la prédiction du modèle SegFormer (overlay + masque)

Lancer :
    streamlit run streamlit_app.py

Variable d'environnement optionnelle :
    API_URL (par défaut "http://localhost:8000")
"""

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

# API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_URL = "https://segmentation-func-api-aphmawergmebcmhz.westeurope-01.azurewebsites.net"
st.set_page_config(page_title="Segmentation U-Net vs SegFormer", layout="wide")

st.title("🧩 Comparaison de segmentation sémantique")
st.caption("U-Net (EfficientNet-B4) vs SegFormer (MiT-B2) — 8 super-catégories Cityscapes")


def b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


@st.cache_data(show_spinner=False, ttl=300)
def get_palette(api_url: str):
    """Récupère la palette {super-catégorie: [r, g, b]} depuis l'API. Retourne None si indisponible."""
    try:
        resp = requests.get(f"{api_url}/classes", timeout=5)
        if resp.ok:
            return resp.json()["palette"]
    except requests.exceptions.RequestException:
        pass
    return None


def render_legend(palette: dict, horizontal: bool = True):
    """Affiche la légende couleur -> super-catégorie. Disposition horizontale (pastilles en ligne) ou verticale."""
    if not palette:
        st.warning("⚠️ Légende indisponible : impossible de contacter l'API.")
        return

    if horizontal:
        items_html = "".join(
            f"<div style='display:flex;align-items:center;gap:6px;margin-right:18px;margin-bottom:6px;'>"
            f"<div style='width:16px;height:16px;border-radius:3px;background-color:rgb({r},{g},{b});"
            f"border:1px solid #888;flex-shrink:0;'></div>"
            f"<span style='font-size:0.85rem;white-space:nowrap;'>{cat}</span></div>"
            for cat, (r, g, b) in palette.items()
        )
        st.markdown(
            f"<div style='display:flex;flex-wrap:wrap;align-items:center;'>{items_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        for cat, (r, g, b) in palette.items():
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<div style='width:16px;height:16px;border-radius:3px;background-color:rgb({r},{g},{b});"
                f"border:1px solid #888;'></div><span>{cat}</span></div>",
                unsafe_allow_html=True,
            )


# --- Barre latérale : paramètres ---
with st.sidebar:
    st.header("Paramètres")
    api_url = st.text_input("URL de l'API FastAPI", value=API_URL)
    alpha = st.slider("Opacité du masque (overlay)", 0.0, 1.0, 0.5, 0.05)
    view_mode = st.radio("Mode d'affichage", ["Overlay sur l'image", "Masque pur"], index=0)

    st.divider()
    st.subheader("Légende des classes")
    render_legend(get_palette(api_url), horizontal=False)


uploaded_file = st.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with st.spinner("Prédiction en cours pour les deux modèles…"):
        try:
            files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type or "image/png")}
            resp = requests.post(f"{api_url}/predict", files=files, params={"alpha": alpha}, timeout=60)
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter l'API à l'adresse {api_url}. Détail : {e}")
            st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"Erreur API ({resp.status_code}) : {detail}")
        st.stop()

    data = resp.json()
    results = data["results"]

    key = "overlay_b64" if view_mode == "Overlay sur l'image" else "mask_b64"

    st.markdown("**Légende des couleurs**")
    render_legend(get_palette(api_url), horizontal=True)
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Image originale")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("🔷 U-Net")
        st.image(b64_to_image(results["unet"][key]), use_container_width=True)

    with col3:
        st.subheader("🔶 SegFormer")
        st.image(b64_to_image(results["segformer"][key]), use_container_width=True)

    st.divider()
    st.subheader("📊 Répartition des classes prédites (%)")

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.markdown("**U-Net**")
        st.bar_chart(results["unet"]["class_distribution"])
    with dist_col2:
        st.markdown("**SegFormer**")
        st.bar_chart(results["segformer"]["class_distribution"])

else:
    st.info("⬆️ Uploadez une image pour lancer la comparaison des deux modèles.")