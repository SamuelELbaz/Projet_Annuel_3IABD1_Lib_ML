import ctypes
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
def _find_c_source() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        p / "model_lineaire.c" for p in [cwd, *cwd.parents]
    ] + [
        p / "model_lineaire" / "model_lineaire.c" for p in [cwd, *cwd.parents]
    ]
    c_src = next((c for c in candidates if c.exists()), None)
    if c_src is None:
        raise FileNotFoundError("Impossible de trouver model_lineaire.c depuis le dossier courant")
    return c_src


def _load_lib():
    c_src = _find_c_source()

    # tmp_so = Path("/tmp") / f"model_lineaire_{int(time.time())}.so"
    # cmd = ["gcc", "-shared", "-fPIC", "-O2", "-o", str(tmp_so), str(c_src), "-lm"]
    # res = subprocess.run(cmd, capture_output=True, text=True)
    # if res.returncode != 0:
    #     raise RuntimeError(f"Compilation C impossible:\n{res.stderr}")

    # lib = ctypes.CDLL(str(tmp_so))

    import tempfile
    tmp_dir = Path(tempfile.gettempdir())
    tmp_so  = tmp_dir / f"model_lineaire_{int(time.time())}.dll"
    
    cmd = ["gcc", "-shared", "-O2", "-o", str(tmp_so), str(c_src), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Compilation C impossible:\n{res.stderr}")

    lib = ctypes.CDLL(str(tmp_so))
    
    # Signatures C (pointeurs 1D, pas de taille max)
    lib.rosenblatt.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # X
        ctypes.POINTER(ctypes.c_int),     # Y
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, nf, nc
        ctypes.c_double, ctypes.c_int,    # alpha, max_ep
        ctypes.POINTER(ctypes.c_double),  # W
        ctypes.c_uint,                    # seed
    ]
    lib.rosenblatt.restype = None
    
    lib.pseudo_inverse.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # X
        ctypes.POINTER(ctypes.c_double),  # Y
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, nf, nc
        ctypes.POINTER(ctypes.c_double),  # W
    ]
    lib.pseudo_inverse.restype = None
    
    return lib


lib = _load_lib()


class LinearModel:
    def __init__(self, layers, learning_rate=0.1, max_ep=2000, classification_algo="rosenblatt", is_regression=False, seed=None):
        self.learning_rate = float(learning_rate)
        self.max_ep = int(max_ep)
        self.input_dim = int(layers[0])
        self.output_dim = int(layers[-1]) if len(layers) > 1 else 1
        self.nf = self.input_dim + 1  # +1 pour le biais
        self.nc = 2 if self.output_dim == 1 else self.output_dim
        self.is_regression = bool(is_regression)
        self.seed = seed  # None = random, int = fixed
        self.w_cls = np.zeros((self.nf, self.nc), dtype=np.float64)
        self.w_reg = np.zeros(self.nf, dtype=np.float64)

    def _x2d(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_dim:
            raise ValueError(f"X doit avoir {self.input_dim} features")
        return X

    def _xb(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])

    def train(self, X, Y):
        X = self._x2d(X)
        Xb = self._xb(X)
        n = Xb.shape[0]

        if self.is_regression:
            y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
            nc = y.shape[1]
            
            X_c = np.ascontiguousarray(Xb, dtype=np.float64)
            Y_c = np.ascontiguousarray(y, dtype=np.float64)
            W_c = np.zeros((self.nf, nc), dtype=np.float64, order="C")
            
            lib.pseudo_inverse(
                X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(n), ctypes.c_int(self.nf), ctypes.c_int(nc),
                W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            
            self.w_reg = W_c[:, 0].copy()
            return float(np.mean((Xb @ self.w_reg - y.ravel()) ** 2))

        # Classification
        y = np.asarray(Y)
        if y.ndim == 2 and y.shape[1] > 1:
            y_idx = np.argmax(y, axis=1).astype(np.int32)
        else:
            yv = y.reshape(-1).astype(np.float64)
            y_idx = (yv > 0).astype(np.int32) if self.nc == 2 else yv.astype(np.int32)
        y_idx = np.clip(y_idx, 0, self.nc - 1)

        X_c = np.ascontiguousarray(Xb, dtype=np.float64)
        Y_c = np.ascontiguousarray(y_idx, dtype=np.int32)
        W_c = np.zeros((self.nf, self.nc), dtype=np.float64, order="C")

        # Seed: si None, utiliser un seed aléatoire différent à chaque appel
        actual_seed = self.seed if self.seed is not None else np.random.randint(0, 2**31)

        lib.rosenblatt(
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(n), ctypes.c_int(self.nf), ctypes.c_int(self.nc),
            ctypes.c_double(self.learning_rate), ctypes.c_int(self.max_ep),
            W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_uint(actual_seed),
        )

        self.w_cls = np.nan_to_num(W_c, nan=0.0, posinf=0.0, neginf=0.0)
        
        scores = Xb @ self.w_cls
        if self.nc == 2 and self.output_dim == 1:
            margins = scores[:, 1] - scores[:, 0]
            target = np.where(y_idx == 1, 1.0, -1.0)
            return float(np.mean((margins - target) ** 2))
        target = np.zeros((n, self.nc), dtype=np.float64)
        target[np.arange(n), y_idx] = 1.0
        return float(np.mean((scores - target) ** 2))

    def forward(self, x):
        xb = self._xb(np.asarray(x, dtype=np.float64).reshape(1, -1))
        if self.is_regression:
            val = (xb @ self.w_reg).item()
            return np.array([val], dtype=np.float64)
        scores = xb @ self.w_cls
        if self.nc == 2 and self.output_dim == 1:
            return np.array([float(scores[0, 1] - scores[0, 0])], dtype=np.float64)
        return scores.ravel()

    def predict_class(self, X):
        scores = self._xb(self._x2d(X)) @ self.w_cls
        if self.nc == 2 and self.output_dim == 1:
            return (scores[:, 1] - scores[:, 0] > 0).astype(int)
        return np.argmax(scores, axis=1)

    def accuracy(self, X, labels):
        labels = np.asarray(labels).astype(int).ravel()
        return float(np.mean(self.predict_class(X) == labels))

    def confusion_matrix(self, X, labels):
        labels = np.asarray(labels).astype(int).ravel()
        pred = self.predict_class(X)
        n = int(max(labels.max(initial=0), pred.max(initial=0)) + 1)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(labels, pred):
            cm[t, p] += 1
        return cm

    def loss(self, X, Y):
        Xb = self._xb(self._x2d(X))
        n = Xb.shape[0]
        scores = Xb @ self.w_cls

        Y = np.asarray(Y).ravel().astype(int)
        target = np.zeros((n, self.nc))
        target[np.arange(n), Y] = 1.0

        return float(np.mean((scores - target) ** 2))  # MSE

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self.input_dim} {self.output_dim} {self.nc} {int(self.is_regression)}\n")
            if self.is_regression:
                f.write(' '.join(map(str, self.w_reg)) + '\n')
            else:
                for row in self.w_cls:
                    f.write(' '.join(map(str, row)) + '\n')

    @classmethod
    def load(cls, path):
        with open(path) as f:
            input_dim, output_dim, nc, is_regression = f.readline().split()
            input_dim, output_dim, nc = int(input_dim), int(output_dim), int(nc)
            is_regression = bool(int(is_regression))
            model = cls([input_dim, output_dim], is_regression=is_regression)
            if is_regression:
                model.w_reg = np.array(list(map(float, f.readline().split())))
            else:
                rows = [list(map(float, f.readline().split())) for _ in range(input_dim + 1)]
                model.w_cls = np.array(rows)
            return model