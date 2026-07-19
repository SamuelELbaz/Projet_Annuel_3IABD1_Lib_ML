import ctypes
import numpy as np
import os

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "..", "lib", "pmc.dll"))

_lib.init_pmc.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # layer_sizes
    ctypes.c_int,                  # nb_sizes
    ctypes.c_double,                # learning_rate
    ctypes.c_int
]
_lib.init_pmc.restype = ctypes.c_void_p  # PMC* opaque

_lib.free_pmc.argtypes = [ctypes.c_void_p]
_lib.free_pmc.restype  = None

_lib.pmc_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_lib.pmc_forward.restype  = ctypes.POINTER(ctypes.c_double)

_lib.pmc_train_all.argtypes = [
    ctypes.c_void_p,               # pmc
    ctypes.POINTER(ctypes.c_double),  # inputs
    ctypes.POINTER(ctypes.c_double),  # answers
    ctypes.c_int,                  # nb_samples
    ctypes.c_int,                  # nb_features
    ctypes.c_int                   # nb_outputs
]
_lib.pmc_train_all.restype = ctypes.c_double

_lib.pmc_loss.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
_lib.pmc_loss.restype = ctypes.c_double

_lib.pmc_accuracy.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
_lib.pmc_accuracy.restype = ctypes.c_double

_lib.pmc_confusion_matrix.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
_lib.pmc_confusion_matrix.restype = None

_lib.pmc_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.pmc_save.restype  = ctypes.c_int

_lib.pmc_load.argtypes = [ctypes.c_char_p]
_lib.pmc_load.restype  = ctypes.c_void_p


def _to_c_double(arr):
    a = np.asarray(arr, dtype=np.float64, order='C')
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a


class PMC:
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.5, is_regression: bool = False):
        sizes_arr = (ctypes.c_int * len(layer_sizes))(*layer_sizes)
        self._pmc = _lib.init_pmc(sizes_arr, len(layer_sizes), learning_rate, int(is_regression))
        if not self._pmc:
            raise RuntimeError("init_pmc a échoué")
        self._nb_outputs = layer_sizes[-1]

    def __del__(self):
        if hasattr(self, "_pmc") and self._pmc:
            _lib.free_pmc(self._pmc)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_ptr, _ = _to_c_double(x)
        out_ptr = _lib.pmc_forward(self._pmc, x_ptr)
        return np.ctypeslib.as_array(out_ptr, shape=(self._nb_outputs,)).copy()

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        nb_samples, nb_features = X.shape
        x_ptr, _x = _to_c_double(X.flatten())
        y_ptr, _y = _to_c_double(y.flatten())
        return _lib.pmc_train_all(
            self._pmc, x_ptr, y_ptr,
            nb_samples, nb_features, self._nb_outputs
        )

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        nb_samples, nb_features = X.shape
        x_ptr, _x = _to_c_double(X.flatten())
        y_ptr, _y = _to_c_double(y.flatten())
        return _lib.pmc_loss(
            self._pmc, x_ptr, y_ptr,
            nb_samples, nb_features, self._nb_outputs
        )

    def accuracy(self, X: np.ndarray, labels: np.ndarray) -> float:
        nb_samples, nb_features = X.shape
        x_ptr, _x  = _to_c_double(X.flatten())
        lb_ptr, _lb = _to_c_double(labels.astype(np.float64))
        return _lib.pmc_accuracy(
            self._pmc, x_ptr, lb_ptr,
            nb_samples, nb_features, self._nb_outputs
        )

    def confusion_matrix(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        nb_samples, nb_features = X.shape
        nb_classes = 2 if self._nb_outputs == 1 else self._nb_outputs
        x_ptr, _x   = _to_c_double(X.flatten())
        lb_ptr, _lb = _to_c_double(labels.astype(np.float64))
        cm = (ctypes.c_int * (nb_classes ** 2))()
        _lib.pmc_confusion_matrix(
            self._pmc, x_ptr, lb_ptr,
            nb_samples, nb_features, self._nb_outputs,
            cm
        )
        return np.ctypeslib.as_array(cm).reshape(
            nb_classes, nb_classes
        ).copy()
    
    def save(self, chemin: str) -> bool:
        result = _lib.pmc_save(self._pmc, chemin.encode('utf-8'))
        return result == 0
    
    @classmethod
    def load(cls, chemin: str) -> 'PMC':
        ptr = _lib.pmc_load(chemin.encode('utf-8'))
        if not ptr:
            raise FileNotFoundError(f"Impossible de charger : {chemin}")

        # Creer un objet PMC sans appeler __init__
        obj = cls.__new__(cls)
        obj._pmc = ptr

        # Recuperer nb_outputs depuis le fichier
        with open(chemin, 'r') as f:
            for line in f:
                if line.startswith('sizes'):
                    sizes = list(map(int, line.split()[1:]))
                    obj._nb_outputs = sizes[-1]
                    break

        return obj
