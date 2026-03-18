import ctypes

lib = ctypes.CDLL("./cmake-build-debug/libProjet_Annuel_3IABD.dll")

print(lib.add(1,5))