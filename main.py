"""
API FastAPI de prédiction de segmentation sémantique.

Expose un modèle U-Net et un modèle SegFormer et renvoie, pour une image donnée :
- le masque colorisé de chaque modèle
- la superposition (overlay) du masque sur l'image originale
- la distribution (%) des super-catégories prédites par chaque modèle

Lancer en local :
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app import model_utils as mu

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement des modèles une seule fois, au démarrage de l'API
    try:
        unet, segformer = mu.load_models()
        models["unet"] = unet
        models["segformer"] = segformer
        print(f"Modèles chargés avec succès sur device={mu.DEVICE}")
    except FileNotFoundError as e:
        # On laisse l'API démarrer pour que /health reste consultable,
        # mais /predict renverra une erreur explicite tant que les poids ne sont pas présents.
        print(f"[ATTENTION] {e}")
    yield
    models.clear()


app = FastAPI(
    title="API de Segmentation Sémantique — U-Net vs SegFormer",
    description="Compare les prédictions de segmentation d'un U-Net (efficientnet-b4) "
    "et d'un SegFormer (mit_b2) entraînés sur Cityscapes (8 super-catégories).",
    version="1.0.0",
    lifespan=lifespan,
)

# Permet à l'app Streamlit d'appeler l'API depuis un autre host/port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(mu.DEVICE),
        "models_loaded": list(models.keys()),
    }


@app.get("/classes")
def classes():
    return {"classes": mu.SUPERCATS, "palette": mu.PALETTE}


def _ensure_models_loaded():
    if "unet" not in models or "segformer" not in models:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modèles non chargés. Vérifiez que 'unet_model.pth' et 'segformer.pth' "
                f"sont présents dans le dossier '{mu.MODELS_DIR}', puis redémarrez l'API."
            ),
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...), alpha: float = 0.5):
    """
    Reçoit une image, exécute U-Net et SegFormer, et renvoie pour chacun :
    - le masque colorisé (mask_b64)
    - la superposition image + masque (overlay_b64)
    - la distribution des classes en %

    Renvoie aussi l'image originale redimensionnée (pour affichage côte à côte côté client).
    """
    _ensure_models_loaded()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier envoyé doit être une image.")

    try:
        raw = await file.read()
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image envoyée.")

    tensor = mu.preprocess_image(image)
    resized_original = image.resize((mu.IMG_SIZE[1], mu.IMG_SIZE[0]))

    response = {
        "original_b64": mu.image_to_base64_png(resized_original),
        "results": {},
    }

    for name, model in [("unet", models["unet"]), ("segformer", models["segformer"])]:
        mask = mu.predict_mask(model, tensor)
        mask_img = mu.colorize_mask(mask)
        overlay_img = mu.overlay_mask_on_image(image, mask, alpha=alpha)

        response["results"][name] = {
            "mask_b64": mu.image_to_base64_png(mask_img),
            "overlay_b64": mu.image_to_base64_png(overlay_img),
            "class_distribution": mu.class_pixel_distribution(mask),
        }

    return response
