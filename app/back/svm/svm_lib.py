import ctypes
import numpy as np
import os

# Charger la bibliothèque partagée
import sys
_nom = "svm.dll" if sys.platform == "win32" else "svm.so"
_lib_path = os.path.join(os.path.dirname(__file__), _nom)
_lib = ctypes.CDLL(_lib_path)

SVM_KERNEL_LINEAR = 0
SVM_HARD_MARGIN = 1e30
SVM_KERNEL_POLY = 1
SVM_KERNEL_RBF = 2

# Définir les types de retour et arguments des fonctions C
_lib.svm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
_lib.svm_create.restype = ctypes.c_void_p

_lib.svm_free.argtypes = [ctypes.c_void_p]
_lib.svm_free.restype = None

_lib.svm_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.svm_train.restype = None

_lib.svm_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_lib.svm_predict.restype = ctypes.c_int

_lib.svm_score.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_lib.svm_score.restype = ctypes.c_double

_lib.svm_n_support_vectors.argtypes = [ctypes.c_void_p]
_lib.svm_n_support_vectors.restype = ctypes.c_int

_lib.svm_norme_dans_espace_noyau.argtypes = [ctypes.c_void_p]
_lib.svm_norme_dans_espace_noyau.restype = ctypes.c_double

_lib.svm_n_iterations_used.argtypes = [ctypes.c_void_p]
_lib.svm_n_iterations_used.restype = ctypes.c_int

_lib.svm_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.svm_save.restype = ctypes.c_int

_lib.svm_load.argtypes = [ctypes.c_char_p]
_lib.svm_load.restype = ctypes.c_void_p

class SVM:
    def __init__(self, kernel='linear', gamma=1.0, degree=3.0, max_iter=200000,
                 C_soft=SVM_HARD_MARGIN):
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.C_soft = C_soft
        self._model = None
        self._X_ptr = None
        self._Y_ptr = None
        self._X_rows = None

    def _get_kernel_type(self):
        if self.kernel == 'linear':
            return SVM_KERNEL_LINEAR
        elif self.kernel == 'poly':
            return SVM_KERNEL_POLY
        elif self.kernel == 'rbf':
            return SVM_KERNEL_RBF
        else:
            raise ValueError(f"Noyau inconnu: {self.kernel}")

    def fit(self, X, Y):
        """
        Entraîne le SVM sur les données X, Y.

        X : array-like de shape (n_samples, n_features)
        Y : array-like de shape (n_samples,) avec valeurs -1 ou 1
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.int32).flatten()

        n_samples, n_features = X.shape

        # Libérer l'ancien modèle si existant
        if self._model is not None:
            _lib.svm_free(self._model)

        # Créer le modèle
        self._model = _lib.svm_create(
            n_samples, n_features,
            self._get_kernel_type(),
            self.gamma, self.degree,
            self.C_soft
        )

        # Préparer X comme tableau de pointeurs (double**)
        self._X_rows = (ctypes.POINTER(ctypes.c_double) * n_samples)()
        for i in range(n_samples):
            self._X_rows[i] = X[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Préparer Y
        self._Y_ptr = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Garder références pour éviter garbage collection
        self._X_data = X
        self._Y_data = Y

        # Entraîner
        _lib.svm_train(self._model, self._X_rows, self._Y_ptr, self.max_iter)

        return self

    def predict(self, X):
        """Prédit les classes pour X."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.zeros(len(X), dtype=np.int32)
        for i, x in enumerate(X):
            x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            predictions[i] = _lib.svm_predict(self._model, x_ptr)

        return predictions

    def decision_function(self, X):
        """Retourne les scores de décision pour X."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = np.zeros(len(X), dtype=np.float64)
        for i, x in enumerate(X):
            x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            scores[i] = _lib.svm_score(self._model, x_ptr)

        return scores

    def accuracy(self, X, Y):
        """Calcule l'accuracy sur X, Y."""
        predictions = self.predict(X)
        Y = np.asarray(Y).flatten()
        return np.mean(predictions == Y)

    def n_support_vectors(self):
        """Retourne le nombre de vecteurs supports."""
        if self._model is None:
            return 0
        return _lib.svm_n_support_vectors(self._model)

    def norme_w(self):
        if self._model is None:
            return 0.0
        return _lib.svm_norme_dans_espace_noyau(self._model)

    def n_iterations_used(self):
        if self._model is None:
            return 0
        return _lib.svm_n_iterations_used(self._model)

    def save(self, path):
        if self._model is None:
            raise RuntimeError("Le modele doit etre entraine (fit) avant d'etre sauvegarde.")
        ret = _lib.svm_save(self._model, path.encode('utf-8'))
        if ret != 0:
            raise IOError(f"Impossible d'ecrire le modele dans {path}")

    @classmethod
    def load(cls, path):
        model_ptr = _lib.svm_load(path.encode('utf-8'))
        if not model_ptr:
            raise IOError(f"Impossible de charger le modele depuis {path}")
        instance = cls()
        instance._model = model_ptr
        return instance

    def __del__(self):
        if self._model is not None:
            _lib.svm_free(self._model)
            self._model = None


class SVM_OVR:
    def __init__(self, n_classes, kernel='linear', gamma=1.0, degree=3.0, max_iter=200000,
                 C_soft=SVM_HARD_MARGIN):
        self.n_classes = n_classes
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.C_soft = C_soft
        self.models = []

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.int32).flatten()

        self.models = []
        for c in range(self.n_classes):
            # Labels binaires: +1 pour classe c, -1 pour toutes les autres
            Y_bin = np.where(Y == c, 1, -1).astype(np.int32)

            model = SVM(
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                max_iter=self.max_iter,
                C_soft=self.C_soft
            )
            model.fit(X, Y_bin)
            self.models.append(model)

        return self

    def predict(self, X):
        """Prédit les classes pour X (argmax des scores)."""
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for c, model in enumerate(self.models):
            norme = model.norme_w()
            norme = norme if norme > 1e-12 else 1.0  # garde-fou division par 0
            scores[:, c] = model.decision_function(X) / norme

        return scores

    def accuracy(self, X, Y):
        """Calcule l'accuracy sur X, Y."""
        predictions = self.predict(X)
        Y = np.asarray(Y).flatten()
        return np.mean(predictions == Y)

    def n_support_vectors(self):
        """Retourne le nombre de vecteurs supports par classifieur."""
        return [m.n_support_vectors() for m in self.models]

    def n_iterations_used(self):
        """Retourne le nombre d'itérations par classifieur."""
        return [m.n_iterations_used() for m in self.models]

    def confusion_matrix(self, X, Y):
        """Calcule la matrice de confusion."""
        predictions = self.predict(X)
        Y = np.asarray(Y).flatten()

        cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int32)
        for true_class, pred_class in zip(Y, predictions):
            cm[true_class, pred_class] += 1

        return cm

    def save(self, path):
        if not self.models:
            raise RuntimeError("Le modele doit etre entraine (fit) avant d'etre sauvegarde.")

        with open(path, "w") as out:
            out.write(f"{self.n_classes} {self.kernel} {self.gamma} {self.degree}\n")

            for model in self.models:
                tmp_path = path + ".tmp"
                model.save(tmp_path)
                with open(tmp_path) as tmp_f:
                    contenu = tmp_f.read()
                os.remove(tmp_path)

                out.write("### CLASSIFIEUR\n")
                out.write(contenu)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            entete = f.readline().split()
            n_classes = int(entete[0])
            kernel = entete[1]
            gamma = float(entete[2])
            degree = float(entete[3])
            contenu = f.read()

        blocs = contenu.split("### CLASSIFIEUR\n")[1:]  # [0] est vide (avant le 1er marqueur)

        instance = cls(n_classes=n_classes, kernel=kernel, gamma=gamma, degree=degree)
        instance.models = []

        for bloc in blocs:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as tmp_f:
                tmp_f.write(bloc)
            instance.models.append(SVM.load(tmp_path))
            os.remove(tmp_path)

        return instance

