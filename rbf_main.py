import ctypes
import numpy as np
from tqdm import tqdm

import os
_here = os.path.dirname(os.path.abspath(__file__))
_dll = os.path.join(_here, "rbf (1).dll")
if not os.path.exists(_dll):
    _dll = os.path.join(_here, "rbf_cpp.so")
lib = ctypes.CDLL(_dll)

c_int = ctypes.c_int; c_uint = ctypes.c_uint
c_double = ctypes.c_double; c_double_p = ctypes.POINTER(c_double)

lib.set_seed.argtypes = [c_uint]; lib.set_seed.restype = None

lib.train_reg.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                          c_double_p, c_int, c_double, c_double_p]
lib.train_reg.restype  = c_double

lib.init_class.argtypes = [c_double_p, c_int, c_int, c_int, c_double,
                           c_double_p, c_double_p]
lib.init_class.restype  = None
lib.train_epoch.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                            c_double_p, c_double]
lib.train_epoch.restype  = c_double

lib.predict.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                        c_double_p, c_int, c_double, c_double_p]
lib.predict.restype  = None

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