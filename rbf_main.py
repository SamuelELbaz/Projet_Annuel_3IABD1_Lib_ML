import os
import ctypes
import platform
import numpy as np
import pandas as pd # type: ignore
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps

_here = os.path.dirname(os.path.abspath(__file__))
_ext = ".dll" if platform.system() == "Windows" else ".so"

ALGOS = {
    "random":   "lib_rbf_km",
    "kmeans++": "lib_rbf_kmpp",
}

c_int = ctypes.c_int; c_uint = ctypes.c_uint
c_double = ctypes.c_double; c_double_p = ctypes.POINTER(c_double)

lib = None

def _declare_signatures(l):
    l.set_seed.argtypes = [c_uint]; l.set_seed.restype = None

    l.train_reg.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                            c_double_p, c_int, c_double, c_double_p]
    l.train_reg.restype  = c_double

    l.init_class.argtypes = [c_double_p, c_int, c_int, c_int, c_double,
                            c_double_p, c_double_p]
    l.init_class.restype  = None

    l.train_epoch.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                            c_double_p, c_double]
    l.train_epoch.restype  = c_double

    l.predict.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                        c_double_p, c_int, c_double, c_double_p]
    l.predict.restype  = None

def set_algo(name):
    global lib
    print(f"Utilisation de l'algorithme : {name}")
    path = os.path.join(_here, ALGOS[name] + _ext)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable")
    lib = ctypes.CDLL(path)
    _declare_signatures(lib)

set_algo("random")

def _p(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(c_double_p)

def one_hot(labels, nb_classes):
    labels = np.asarray(labels)
    Y = np.zeros((len(labels), nb_classes))
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

def confusion_matrix(y_true, y_pred, nb_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((nb_classes, nb_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)   # ligne = vrai, colonne = prédit
    return cm

def set_seed(s):
    lib.set_seed(int(s))

class RBF:
    def __init__(self, nb_centres, gamma):
        self.K = nb_centres
        self.gamma = gamma
        self.centres = self.W = None
        self.dim = self.nb_classes = None

    def train_reg(self, X, Y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        n, dim = X.shape
        nb_classes = Y.shape[1]
        centres_out = np.zeros(self.K * dim)
        W_out       = np.zeros(nb_classes * self.K)
        _, Xp = _p(X.ravel()); _, Yp = _p(Y.ravel())
        mse = lib.train_reg(Xp, n, dim, Yp, nb_classes,
                            centres_out.ctypes.data_as(c_double_p), self.K, self.gamma,
                            W_out.ctypes.data_as(c_double_p))
        self.centres = centres_out.reshape(self.K, dim)
        self.W = W_out.reshape(nb_classes, self.K)
        self.dim = dim; self.nb_classes = nb_classes
        return mse

    train = train_reg

    def init_classif(self, X, nb_classes):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, dim = X.shape
        self.dim, self.nb_classes = dim, nb_classes
        self.centres = np.zeros(self.K * dim)
        self._phi    = np.zeros(n * self.K)
        _, Xp = _p(X.ravel())
        lib.init_class(Xp, n, dim, self.K, self.gamma,
                       self.centres.ctypes.data_as(c_double_p),
                       self._phi.ctypes.data_as(c_double_p))
        self.centres = self.centres.reshape(self.K, dim)
        self.W = np.zeros(nb_classes * self.K)
        self._n = n

    def train_epoch(self, Y_onehot, learning_rate=0.05):
        Y = np.ascontiguousarray(Y_onehot, dtype=np.float64)
        _, Yp = _p(Y.ravel())
        return lib.train_epoch(self._phi.ctypes.data_as(c_double_p),
                               self._n, self.K, Yp, self.nb_classes,
                               self.W.ctypes.data_as(c_double_p), learning_rate)

    def predict_scores(self, X):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        n = X.shape[0]
        scores = np.zeros(n * self.nb_classes)
        _, Xp = _p(X.ravel())
        _, cp = _p(self.centres.ravel())
        _, wp = _p(self.W.ravel())          # W.ravel() : marche a plat ou en 2D
        lib.predict(Xp, n, self.dim, cp, self.K, wp, self.nb_classes,
                    self.gamma, scores.ctypes.data_as(c_double_p))
        return scores.reshape(n, self.nb_classes)

    def forward(self, pt):
        return self.predict_scores(np.atleast_2d(pt))[0]

    def predict(self, X):
        return self.predict_scores(X).argmax(axis=1)
    
    def sauvegarder(self, chemin, classes=None):
            if self.centres is None or self.W is None:
                raise RuntimeError("modèle non entrainé")
            np.savez(chemin,
                    centres=self.centres,
                    W=np.asarray(self.W).reshape(self.nb_classes, self.K),
                    gamma=np.float64(self.gamma), K=self.K,
                    dim=self.dim, nb_classes=self.nb_classes,
                    classes=np.array(classes if classes is not None else np.arange(self.nb_classes)))

    @classmethod
    def charger(cls, chemin):
        d = np.load(chemin, allow_pickle=True)
        m = cls(nb_centres=int(d["K"]), gamma=float(d["gamma"]))
        m.centres    = d["centres"]
        m.W          = d["W"]
        m.dim        = int(d["dim"])
        m.nb_classes = int(d["nb_classes"])
        m.classes    = list(d["classes"])
        return m

## Hmm

def rbf_classification_training( 
                                nb_epoch:int
                                ,learning_rate:float
                                ,nb_centre:int
                                ,gamma:float
                                ,seeds:list[int]
                                ,X_train, X_val
                                ,y_train, y_val
                                )->list[list|list[list]] :

    Y_train_oh = one_hot(y_train, 3)
    Y_val_oh   = one_hot(y_val, 3)


    all_accs, all_train_loss, all_val_loss = [], [], []
        
    for seed in seeds :
        set_seed(seed)
        model = RBF(nb_centres=nb_centre, gamma=gamma)
        model.init_classif(X_train, 3)

        train_losses, val_losses, accs = [], [], []

        pbar = tqdm(range(nb_epoch), desc=f"seed={seed}")
        for epoch in pbar:
            train_losses.append(model.train_epoch(Y_train_oh, learning_rate))

            scores_val = model.predict_scores(X_val)
            val_losses.append(np.mean((scores_val - Y_val_oh) ** 2))

            accs.append(np.mean(model.predict(X_val) == y_val))

            pbar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}", acc=f"{accs[-1]:.3f}")
        
        all_train_loss.append(train_losses)
        all_val_loss.append(val_losses)
        all_accs.append(accs)

    all_train_loss = np.array(all_train_loss)
    all_val_loss   = np.array(all_val_loss)
    all_accs       = np.array(all_accs)

    return {
        "model": model,
        "train_loss_mean": all_train_loss.mean(axis=0),
        "train_loss_std":  all_train_loss.std(axis=0),
        "val_loss_mean":   all_val_loss.mean(axis=0),
        "val_loss_std":    all_val_loss.std(axis=0),
        "accs_mean":       all_accs.mean(axis=0),
        "accs_std":        all_accs.std(axis=0),
    }

def rbf_train_export(
                    nb_epoch:int
                    ,learning_rate:float
                    ,nb_centre:int
                    ,gamma:float
                    ,seeds:list[int]
                    ,X_train, X_val
                    ,y_train, y_val
                    ):
    
    Y_train_oh = one_hot(y_train, 3)

    best_model, best_seed, best_acc = None, None, -1.0

    for seed in seeds:
        set_seed(seed)
        model = RBF(nb_centres=nb_centre, gamma=gamma)
        model.init_classif(X_train, 3)

        for _ in tqdm(range(nb_epoch), desc=f"seed={seed}"):
            model.train_epoch(Y_train_oh, learning_rate)

        acc = np.mean(model.predict(X_val) == y_val)
        print(f"seed={seed} | acc val={acc:.4f}")
        if acc > best_acc:
            best_acc, best_seed, best_model = acc, seed, model

    print(f"-> retenu seed={best_seed} | acc val={best_acc:.4f}")
    return best_model


# Prepa data set
def data_set_separation(seed, nb_img_per_classes, ratios):
    #np.random.seed(12)
    #NB_IMG_PER_CLASSES = 1000

    base = Path("data")
    classes = ["compost", "dechets_chimiques", "recyclable"]

    rows = []
    for cls in classes:
        images = sorted((base / cls).glob("*.jpg"))[:nb_img_per_classes]
        for img_path in images:
            rows.append({
                "path": str(img_path),
                "nom_image": img_path.name,
                "classe": cls
            })

    df = pd.DataFrame(rows)

    def assign_split(n, ratios):
        labels = np.array(["train"] * n)
        idx = np.random.permutation(n)
        t1 = int(n * ratios[0])
        t2 = int(n * (ratios[0] + ratios[1]))
        labels[idx[t1:t2]] = "val"
        labels[idx[t2:]] = "test"
        return labels

    df["cat"] = df.groupby("classe")["path"].transform(lambda x: assign_split(len(x), ratios))

    print(f"Total : {len(df)} images")
    print(df.groupby("classe")["cat"].value_counts())
    df.head()
    
    return df

# Transform (resolution + couleur)
CLASSES = ["compost", "dechets_chimiques", "recyclable"]

def data_set_transform(df, img_size, color="gray", verbose=True):
    """
    img_size : dimension 
    color    : "gray", "rgb"
    """
    mode = "L" if color == "gray" else "RGB"
    size = (img_size, img_size)

    def load_image(path):
        try:
            img = Image.open(path).convert(mode)
            img = ImageOps.fit(img, size, method=Image.LANCZOS)
            return np.array(img, dtype=np.float32).flatten() / 255.0
        except (OSError, IOError):
            return None

    df = df.copy()
    df["pixels"] = [load_image(p) for p in tqdm(df["path"], desc=f"{img_size}px {color}")]

    n_before = len(df)
    df = df.dropna(subset=["pixels"]).reset_index(drop=True)

    label_map = {cls: i for i, cls in enumerate(CLASSES)}

    def split(cat):
        sub = df[df["cat"] == cat]
        X = np.stack(sub["pixels"].values)
        y = sub["classe"].map(label_map).values
        return X, y

    X_train, y_train = split("train")
    X_val,   y_val   = split("val")
    X_test,  y_test  = split("test")

    if verbose:
        print(f"Images corrompues supprimées : {n_before - len(df)}")
        print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
        print(f"dim = {X_train.shape[1]}  |  labels : {label_map}")

    return X_train, X_val, X_test, y_train, y_val, y_test