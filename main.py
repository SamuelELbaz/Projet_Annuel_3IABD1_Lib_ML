import ctypes
from PIL import Image

lib = ctypes.CDLL("./cmake-build-debug/libProjet_Annuel_3IABD.dll")

c_int = ctypes.c_int
c_double = ctypes.c_double
c_void_p = ctypes.c_void_p

c_double_p = ctypes.POINTER(c_double)
c_double_p_p = ctypes.POINTER(c_double_p)
c_int_p = ctypes.POINTER(c_int)

lib.init_pmc.argtypes = [c_int_p, c_int, c_double]
lib.init_pmc.restype = c_void_p

lib.train.argtypes = [c_void_p, c_double_p, c_double_p, c_int, c_int, c_int]
lib.train.restype = c_double

lib.propagation.argtypes = [c_void_p, c_double_p_p, c_int, c_double_p_p]
lib.propagation.restype = None

lib.predict.argtypes = [c_void_p, c_double_p, c_int, c_int, c_int]
lib.predict.restype = c_double_p

lib.free_p.argtypes = [c_double_p]
lib.free_p.restype = None

lib.free_pmc.argtypes = [c_void_p]
lib.free_pmc.restype = None

# Conversion Image -> vecteur normalisé de pixel (pour avoir des valeurs entre 0 et 1 (sympa pour notre sigmoid))
def image_to_vector(path):
    img = Image.open(path).convert("RGB")

    return [
        c / 255.0
        for pixel in img.getdata()
        for c in pixel
    ]

# --------------------------------------------------------------------



layers = (c_int * 3)(2, 2, 1)
model = lib.init_pmc(layers, 3, 1)

nb_sample = 4
nb_data_per_sample = 2

input = (c_double * (nb_sample * nb_data_per_sample))(
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
)

y_true = (c_double * nb_sample)(
    0.0,
    1.0,
    1.0,
    0.0
)

epochs = 100_000
losses = []

for epoch in range(epochs):
    loss = lib.train(model, input, y_true, 4, 2, 1)
    losses.append(loss)

    if epoch % 1000 == 0:
        print(epoch, loss)

pred_ptr = lib.predict(model, input, 4, 2, 1)

for i in range(4):
    in0 = input[i * 2]
    in1 = input[i * 2 + 1]
    y = pred_ptr[i]
    print(f"[{in0}, {in1}] -> {y}")

lib.free_p(pred_ptr)
lib.free_pmc(model)














