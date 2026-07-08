# Import


import numpy as np
import pandas as pd # type: ignore
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from rbf_main import RBF, one_hot, set_seed, rbf_classification_training, confusion_matrix


## Separation du dataSet


#np.random.seed(4)
NB_IMG_PER_CLASSES = 1000

base = Path("data")
classes = ["compost", "dechets_chimiques", "recyclable"]

rows = []
for cls in classes:
    images = sorted((base / cls).glob("*.jpg"))[:NB_IMG_PER_CLASSES]
    for img_path in images:
        rows.append({
            "path": str(img_path),
            "nom_image": img_path.name,
            "classe": cls
        })

df = pd.DataFrame(rows)

def assign_split(n, ratios=(0.70, 0.28, 0.02)):
    labels = np.array(["train"] * n)
    idx = np.random.permutation(n)
    t1 = int(n * ratios[0])
    t2 = int(n * (ratios[0] + ratios[1]))
    labels[idx[t1:t2]] = "val"
    labels[idx[t2:]] = "test"
    return labels

df["cat"] = df.groupby("classe")["path"].transform(lambda x: assign_split(len(x)))

print(f"Total : {len(df)} images")
print(df.groupby("classe")["cat"].value_counts())
df.head()


## Transformation des données (redimension + gray lvl)


IMG = 24
IMG_SIZE = (IMG, IMG)

def load_image(path):
    try:
        img = Image.open(path).convert("L")
        img = img.resize(IMG_SIZE)
        return np.array(img, dtype=np.float32).flatten() / 255.0
    except (OSError, IOError):
        return None

df["pixels"] = [load_image(p) for p in tqdm(df["path"], desc="Chargement")]

n_before = len(df)
df = df.dropna(subset=["pixels"]).reset_index(drop=True)
print(f"Images corrompues supprimées : {n_before - len(df)}")

X_train = np.stack(df[df["cat"] == "train"]["pixels"].values)
X_val   = np.stack(df[df["cat"] == "val"]["pixels"].values)
X_test  = np.stack(df[df["cat"] == "test"]["pixels"].values)

label_map = {cls: i for i, cls in enumerate(classes)}
y_train = df[df["cat"] == "train"]["classe"].map(label_map).values
y_val   = df[df["cat"] == "val"]["classe"].map(label_map).values
y_test  = df[df["cat"] == "test"]["classe"].map(label_map).values

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"Labels:  {label_map}")


## XP

nb_epoch = 100
seeds_list = [
    [6, 8, 12, 48, 75]
    ,[1, 5, 142, 24, 12]
    ,[3, 8, 9]
]

gammas = [0.01, 0.05, 0.1, 0.5, 0.8]

for g in gammas :
    print(f"Batch gamma = {g}")

    training_output = rbf_classification_training(
            nb_epoch=nb_epoch
            ,learning_rate=.1
            ,nb_centre=100
            ,gamma=g
            ,seeds=seeds_list[2]
            ,X_train=X_train, X_val=X_val
            ,y_train=y_train, y_val=y_val
            )

    model               = training_output["model"]
    train_loss_mean     = training_output["train_loss_mean"]
    train_loss_std      = training_output["train_loss_std"]
    val_loss_mean       = training_output["val_loss_mean"]
    val_loss_std        = training_output["val_loss_std"]
    accs_mean           = training_output["accs_mean"]
    accs_std            = training_output["accs_std"]


    # PLOT
    epochs = np.arange(nb_epoch)

    # LOSSES
    plt.figure(0)
    plt.plot(epochs, train_loss_mean, label=f"Moyenne Train Losses - {g}")
    plt.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.3, label="Ecart-type Train Losses")

    plt.plot(epochs, val_loss_mean, label=f"Moyenne Validation Losses - {g}")
    plt.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.3, label="Ecart-type Validation Losses")

    plt.xlabel("epoch"); plt.ylabel("Losses_Mean")
    plt.title("Moyenne de l'erreur au cours des epoch"); plt.legend()

    # ACCURACY
    plt.figure(11)
    plt.plot(epochs, accs_mean, label=f"Moyenne d'accuracy - {g}")
    plt.fill_between(epochs, accs_mean - accs_std, accs_mean + accs_std, alpha=0.3, label="Ecart-type")
    plt.xlabel("epoch"); plt.ylabel("Accuracy_Mean")
    plt.title("Accuracy validation"); plt.legend(); plt.grid(True)

# CM
y_pred = model.predict(X_val)
cm     = confusion_matrix(y_val, y_pred, 3)
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(classes)
ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
ax.set_title(f"Matrice de confusion (Val)\n Accuracy = {accs_mean[-1]*100:.0f}%")
thresh = cm.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.colorbar(im); plt.tight_layout()

plt.show()
