import ctypes
import numpy as np

import os
_here = os.path.dirname(os.path.abspath(__file__))
_dll = os.path.join(_here, "rbf2.dll")
if not os.path.exists(_dll):
    _dll = os.path.join(_here, "rbf_linux.so")   # fallback test
lib = ctypes.CDLL(_dll)

c_int = ctypes.c_int; c_uint = ctypes.c_uint
c_double = ctypes.c_double; c_double_p = ctypes.POINTER(c_double)

lib.set_seed.argtypes = [c_uint]; lib.set_seed.restype = None
lib.train.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int,
                      c_double_p, c_int, c_double, c_double_p]
lib.train.restype  = c_double
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

def set_seed(s):
    lib.set_seed(int(s))

class RBF:
    def __init__(self, nb_centres, gamma):
        self.K = nb_centres
        self.gamma = gamma
        self.centres = self.W = None
        self.dim = self.nb_classes = None

    def train(self, X, Y_onehot):
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y_onehot, dtype=np.float64)
        n, dim = X.shape
        nb_classes = Y.shape[1]
        centres_out = np.zeros(self.K * dim)
        W_out       = np.zeros(nb_classes * self.K)
        _, Xp = _p(X.ravel()); _, Yp = _p(Y.ravel())
        mse = lib.train(Xp, n, dim, Yp, nb_classes,
                        centres_out.ctypes.data_as(c_double_p), self.K, self.gamma,
                        W_out.ctypes.data_as(c_double_p))
        self.centres = centres_out.reshape(self.K, dim)
        self.W = W_out.reshape(nb_classes, self.K)
        self.dim = dim; self.nb_classes = nb_classes
        return mse

    fit = train

    def predict_scores(self, X):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        n = X.shape[0]
        scores = np.zeros(n * self.nb_classes)
        _, Xp = _p(X.ravel())
        _, cp = _p(self.centres.ravel())
        _, wp = _p(self.W.ravel())
        lib.predict(Xp, n, self.dim, cp, self.K, wp, self.nb_classes,
                    self.gamma, scores.ctypes.data_as(c_double_p))
        return scores.reshape(n, self.nb_classes)

    def forward(self, pt):
        return self.predict_scores(np.atleast_2d(pt))[0]

    def predict(self, X):
        return self.predict_scores(X).argmax(axis=1)
