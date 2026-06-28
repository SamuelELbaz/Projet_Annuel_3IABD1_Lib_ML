import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset.csv"

MODEL_C = BASE_DIR / "model_lineaire" / "model_lineaire.c"
MODEL_SO = BASE_DIR / "model_lineaire" / "model_lineaire.so"

RBF_DIR = BASE_DIR / "rbf"
RBF_DLL = RBF_DIR / "rbf.dll"
RBF_ALT_DLL = RBF_DIR / "rbf2.dll"

MAX_FEAT = 8
MAX_CLS = 3
MAX_SAMP = 4500

LABELS = {0: "Compost", 1: "Chimique", 2: "Recyclable"}
MODEL_UNAVAILABLE = "Indisponible"

app = Flask(__name__)


def _prepare_rbf_dll_name():
    if sys.platform == "win32":
        if not RBF_DLL.exists() and RBF_ALT_DLL.exists():
            RBF_DLL.write_bytes(RBF_ALT_DLL.read_bytes())
        return

    if RBF_DLL.exists() and (RBF_DIR / "rbf_linux.so").exists():
        try:
            RBF_DLL.unlink()
        except OSError:
            pass


def _grid_means(channel: np.ndarray, g: int = 3) -> list[float]:
    h, w = channel.shape
    ys = np.linspace(0, h, g + 1).astype(int)
    xs = np.linspace(0, w, g + 1).astype(int)
    return [
        float(channel[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean())
        for i in range(g)
        for j in range(g)
    ]


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img.thumbnail((224, 224), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    grid = _grid_means(r, 3) + _grid_means(g, 3) + _grid_means(b, 3)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)

    return np.array(
        [float(r.mean()), float(g.mean()), float(b.mean())] + grid + [float(grad.mean()), float(grad.std())],
        dtype=np.float64,
    )


def load_linear_lib() -> ctypes.CDLL:
    if not MODEL_SO.exists():
        cmd = ["gcc", "-shared", "-fPIC", "-DNO_MAIN", "-O2", "-o", str(MODEL_SO), str(MODEL_C), "-lm"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Compilation model_lineaire impossible: {res.stderr}")

    lib = ctypes.CDLL(str(MODEL_SO))
    lib.rosenblatt.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rosenblatt.restype = None
    return lib


def train_linear_weights(lib: ctypes.CDLL) -> np.ndarray:
    df = pd.read_csv(CSV_PATH)

    x = df[["r_mean", "g_mean", "b_mean"]].to_numpy(np.float64) / 255.0
    y = df["label"].to_numpy(np.int32)

    n, nf_no_bias = x.shape
    x_bias = np.hstack([np.ones((n, 1), dtype=np.float64), x])
    nf = nf_no_bias + 1
    nc = int(y.max()) + 1

    if n > MAX_SAMP:
        raise ValueError("Dataset trop grand pour MAX_SAMP")

    x_c = np.zeros((MAX_SAMP, MAX_FEAT), dtype=np.float64, order="C")
    y_c = np.zeros(MAX_SAMP, dtype=np.int32)
    w_c = np.zeros((MAX_FEAT, MAX_CLS), dtype=np.float64, order="C")

    x_c[:n, :nf] = x_bias
    y_c[:n] = y

    lib.rosenblatt(
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n),
        ctypes.c_int(nf),
        ctypes.c_int(nc),
        ctypes.c_double(0.1),
        ctypes.c_int(3000),
        w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return w_c[:nf, :nc].copy()


def load_pmc_model():
    sys.path.insert(0, str(BASE_DIR / "pmc"))
    from pmc import PMC

    df = pd.read_csv(CSV_PATH)
    x = df[["r_mean", "g_mean", "b_mean"]].to_numpy(np.float64) / 255.0
    y_idx = df["label"].to_numpy(np.int32)
    y = np.eye(3, dtype=np.float64)[y_idx]

    model = PMC(layer_sizes=[3, 32, 16, 3], learning_rate=0.05, is_regression=False)
    for _ in range(200):
        model.train(x, y)
    return model


def load_rbf_model():
    _prepare_rbf_dll_name()
    sys.path.insert(0, str(RBF_DIR))
    from rbf_main import RBF, one_hot, set_seed

    df = pd.read_csv(CSV_PATH)
    grid_cols = [f"r_{i}" for i in range(9)] + [f"g_{i}" for i in range(9)] + [f"b_{i}" for i in range(9)]
    x_grid = df[grid_cols].to_numpy(np.float64) / 255.0
    x_grad = df[["grad_mean", "grad_std"]].to_numpy(np.float64)
    x = np.hstack([x_grid, x_grad])
    y = df["label"].to_numpy(np.int32)

    set_seed(42)
    model = RBF(nb_centres=100, gamma=0.5)
    model.train(x, one_hot(y, 3))
    return model


def _safe_load(name: str, loader):
    try:
        return loader(), None
    except Exception as exc:
        return None, f"{name}: {exc}"


MODEL_ERRORS: list[str] = []

LINEAR_LIB, err = _safe_load("Linear lib", load_linear_lib)
if err:
    MODEL_ERRORS.append(err)

LINEAR_W, err = _safe_load("Linear weights", lambda: train_linear_weights(LINEAR_LIB) if LINEAR_LIB is not None else None)
if err:
    MODEL_ERRORS.append(err)

PMC_MODEL, err = _safe_load("PMC", load_pmc_model)
if err:
    MODEL_ERRORS.append(err)

RBF_MODEL, err = _safe_load("RBF", load_rbf_model)
if err:
    MODEL_ERRORS.append(err)


def predict_all(features: np.ndarray) -> dict[str, str]:
    out: dict[str, str] = {"linear": MODEL_UNAVAILABLE, "pmc": MODEL_UNAVAILABLE, "rbf": MODEL_UNAVAILABLE}

    if LINEAR_W is not None:
        x_lin = np.hstack([[1.0], features[:3]])
        pred_lin = int(np.argmax(x_lin @ LINEAR_W))
        out["linear"] = LABELS[pred_lin]

    if PMC_MODEL is not None:
        pred_pmc = int(np.argmax(PMC_MODEL.forward(features[:3])))
        out["pmc"] = LABELS[pred_pmc]

    if RBF_MODEL is not None:
        pred_rbf = int(np.argmax(RBF_MODEL.forward(features[3:32])))
        out["rbf"] = LABELS[pred_rbf]

    return out


@app.get("/")
def home():
    error = None
    if MODEL_ERRORS:
        error = "Certains modèles sont indisponibles: " + " | ".join(MODEL_ERRORS)
    return render_template("index.html", result=None, error=error)


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", result=None, error="Choisis une image.")

    try:
        img = Image.open(file.stream)
        feats = preprocess(img)
        result = predict_all(feats)
        return render_template("index.html", result=result, error=None)
    except Exception as exc:
        return render_template("index.html", result=None, error=f"Erreur: {exc}")


if __name__ == "__main__":
    app.run(debug=True)