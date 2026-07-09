"""
Utilitaires de chargement de modèles et de traitement d'images
pour l'API de segmentation sémantique (U-Net vs SegFormer).

Reprend la pipeline du notebook baseline :
- 8 super-catégories (void, flat, construction, object, nature, sky, human, vehicle)
- Resize (256, 512) + normalisation ImageNet
- U-Net (encoder efficientnet-b4) et SegFormer (encoder mit_b2) via segmentation_models_pytorch
"""

import io
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from azure.storage.blob import BlobServiceClient

# ----------------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------------

NUM_CLASSES = 8
IMG_SIZE = (256, 512)  # (H, W) — identique au notebook d'entraînement

# Ordre identique au dict `cats` du notebook -> index figé.
SUPERCATS = [
    "void",
    "flat",
    "construction",
    "object",
    "nature",
    "sky",
    "human",
    "vehicle",
]

# Palette de couleurs (RGB) pour la visualisation des masques, une couleur par super-catégorie.
PALETTE = {
    "void": (0, 0, 0), # Noir : fond / classe ignorée
    "flat": (128, 64, 128), # Violet : route, trottoir et surfaces planes
    "construction": (70, 70, 70), # Gris foncé : bâtiments, murs, clôtures
    "object": (153, 153, 153), # Gris clair : poteaux, panneaux, feux de circulation
    "nature": (107, 142, 35), # Vert olive : végétation et terrain
    "sky": (70, 130, 180), # Bleu : ciel
    "human": (220, 20, 60), # Rouge : piétons et cyclistes
    "vehicle": (0, 0, 142), # Bleu foncé : voitures, bus, camions, motos, vélos
}

# Tableau (NUM_CLASSES, 3) pour un mapping rapide index -> couleur
COLOR_ARRAY = np.array([PALETTE[c] for c in SUPERCATS], dtype=np.uint8)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "test_model50"))
UNET_WEIGHTS = MODELS_DIR / "unet_model.pth"
SEGFORMER_WEIGHTS = MODELS_DIR / "segformer.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mêmes transformations que pour la validation/inférence dans le notebook
IMG_TRANSFORM = T.Compose(
    [
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Chargement des modèles

def build_unet() -> torch.nn.Module:
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None, #les poids ImageNet pas nécessaires,charge le checkpoint entraîné
        in_channels=3,
        classes=NUM_CLASSES,
    )


def build_segformer() -> torch.nn.Module:
    return smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_segmentation_channels=256,
    )


def _load_checkpoint(model: torch.nn.Module, weights_path: Path) -> torch.nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fichier de poids introuvable : {weights_path}. "
            f"Placez 'unet_model.pth' et 'segformer.pth' dans le dossier '{MODELS_DIR}'."
        )
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def download_models_from_blob():
    conn_str = os.environ.get("MODEL_BLOB_CONNECTION_STRING")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER", "modelweights")

    if not conn_str:
        print("[ATTENTION] AZURE_STORAGE_CONNECTION_STRING non définie.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container = blob_service.get_container_client(container_name)

    for blob_name, local_path in [
        ("unet_model.pth", UNET_WEIGHTS),
        ("segformer.pth", SEGFORMER_WEIGHTS),
    ]:
        if local_path.exists():
            continue

        print(f"Téléchargement de {blob_name} depuis Azure Blob...")
        blob_client = container.get_blob_client(blob_name)

        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
            
def load_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Charge les deux modèles entraînés."""
    download_models_from_blob()

    unet = _load_checkpoint(build_unet(), UNET_WEIGHTS)
    segformer = _load_checkpoint(build_segformer(), SEGFORMER_WEIGHTS)
    return unet, segformer


# Pré-traitement / Inférence / Post-traitement

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL.Image (RGB) -> tensor (1, 3, H, W) prêt pour le modèle."""
    image = image.convert("RGB")
    tensor = IMG_TRANSFORM(image).unsqueeze(0)
    return tensor.to(DEVICE)


@torch.no_grad()
def predict_mask(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Renvoie le masque prédit (H, W) avec les index de classes (0..7)."""
    model.eval()
    logits = model(tensor)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Masque d'index (H, W) -> image RGB colorée selon la palette des super-catégories."""
    color_mask = COLOR_ARRAY[mask]  # (H, W, 3)
    return Image.fromarray(color_mask, mode="RGB")


def overlay_mask_on_image(original: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superpose le masque colorisé sur l'image originale (redimensionnée à IMG_SIZE)."""
    resized_original = original.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    color_mask = colorize_mask(mask)
    blended = Image.blend(resized_original, color_mask, alpha=alpha)
    return blended


def image_to_base64_png(image: Image.Image) -> str:
    import base64

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def class_pixel_distribution(mask: np.ndarray) -> dict:
    """Pourcentage de pixels par super-catégorie, utile pour la comparaison."""
    total = mask.size
    counts = {cat: 0 for cat in SUPERCATS}
    values, occurrences = np.unique(mask, return_counts=True)
    for v, c in zip(values, occurrences):
        counts[SUPERCATS[int(v)]] = round(100 * c / total, 2)
    return counts
