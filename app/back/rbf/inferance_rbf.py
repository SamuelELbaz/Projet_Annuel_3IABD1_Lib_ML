# inference_rbf.py — autonome, ne dépend que de numpy + Pillow
import numpy as np
from PIL import Image, ImageOps

_d       = np.load("rbf_model.npz", allow_pickle=True)
CENTRES  = _d["centres"]
W        = _d["W"]
GAMMA    = float(_d["gamma"])
CLASSES  = list(_d["classes"])
IMG_SIZE = int(np.sqrt(CENTRES.shape[1]))

def _vectoriser(chemin_image):
    """Reproduit EXACTEMENT le preprocessing d'entrainement."""
    img = Image.open(chemin_image).convert("L")
    img = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), method=Image.LANCZOS)
    return np.asarray(img, dtype=np.float64).flatten() / 255.0

def _scores(X):
    X  = np.atleast_2d(X)
    d2 = ((X**2).sum(1)[:, None] + (CENTRES**2).sum(1)[None, :]
          - 2 * X @ CENTRES.T)
    return np.exp(-GAMMA * np.maximum(d2, 0)) @ W.T

def predire(chemin_image):
    s = _scores(_vectoriser(chemin_image))[0]
    return {"classe": CLASSES[int(s.argmax())],
            "scores": {c: float(v) for c, v in zip(CLASSES, s)}}

if __name__ == "__main__":
    import sys
    print(predire(sys.argv[1]))