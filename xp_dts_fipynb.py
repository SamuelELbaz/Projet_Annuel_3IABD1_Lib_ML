# Import


import numpy as np
import pandas as pd # type: ignore
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from rbf_main import RBF, one_hot, set_seed


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

def assign_split(n, ratios=(0.70, 0.15, 0.15)):
    labels = np.array(["train"] * n)
    idx = np.random.permutation(n)
    t1 = int(n * ratios[0])
    t2 = int(n * (ratios[0] + ratios[1]))
    labels[idx[t1:t2]] = "val"
    labels[idx[t2:]] = "test"
    return labels

df["cat"] = df.groupby("classe")["path"] \
              .transform(lambda x: assign_split(len(x)))

print(f"Total : {len(df)} images")
print(df.groupby("classe")["cat"].value_counts())
df.head()


## Transformation des données (redimension + gray lvl)


IMG = 16
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

#pcts = np.arange(10, 50, 10)
#list_nb_centre = (pcts / 100 * NB_IMG_PER_CLASSES * 3).astype(int).tolist()

#list_nb_centre = np.arange(100, 600, 100)
list_nb_centre = 100
seed_list = [6, 8, 12, 48, 75]
nb_seed = len(seed_list)

nb_epoch = 200

all_accs, all_train_loss, all_val_loss = [], [], []

for seed in seed_list :
    set_seed(seed)
    model = RBF(nb_centres=list_nb_centre, gamma=.5)
    model.init_classif(X_train, 3)

    Y_train_oh = one_hot(y_train, 3)
    Y_val_oh   = one_hot(y_val, 3)

    train_losses, val_losses, accs = [], [], []

    for epoch in tqdm(range(nb_epoch)):
        train_losses.append(model.train_epoch(Y_train_oh, learning_rate=.01))

        scores_val = model.predict_scores(X_val)
        val_losses.append(np.mean((scores_val - Y_val_oh) ** 2))

        accs.append(np.mean(model.predict(X_val) == y_val))

        if epoch % int(nb_epoch / 2) == 0:
            print(f"epoch {epoch}: train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}  acc={accs[-1]:.4f}")

    # Liste des losses par seed
    all_train_loss.append(train_losses)
    all_val_loss.append(val_losses)

    # Liste des accuracies par seed
    '''
    all_accs.append(accs)
    plt.figure(0)
    plt.plot(accs)
    '''

# PLOT
epochs = np.arange(nb_epoch)

# LOSSES
all_train_loss = np.array(all_train_loss)
all_train_loss_mean = np.mean(all_train_loss, axis=0)
all_train_loss_std = np.std(all_train_loss)

all_val_loss = np.array(all_val_loss)
all_val_loss_mean = np.mean(all_train_loss, axis=0)
all_val_loss_std = np.std(all_train_loss)

plt.figure(00)
plt.plot(epochs, all_train_loss_mean)
plt.fill_between(epochs, all_train_loss_mean - all_val_loss_std, all_val_loss_mean + all_val_loss_std, alpha=0.3)

plt.plot(epochs, all_val_loss_mean)
plt.fill_between(epochs, all_val_loss_mean - all_val_loss_std, all_val_loss_mean + all_val_loss_std, alpha=0.3)

plt.xlabel("epoch"); plt.ylabel("accuracy");
plt.title("Moyenne de l'erreur au cours des epoch")


# ACCURACY
# Affichage de chaque Courbe d'acc
plt.figure(10)
plt.xlabel("epoch"); plt.ylabel("accuracy");
plt.title("Accuracy validation"); plt.grid(True);

# Affichage d'une courbe de moyenne des accuracy + std (ecart type (standard deviation))
all_accs  = np.array(all_accs)
accs_mean = np.mean(all_accs, axis=0)
accs_std  = np.std(all_accs, axis=0)

plt.figure(11)
plt.plot(epochs, accs_mean)
plt.xlabel("epoch"); plt.ylabel("accuracy_mean");
plt.fill_between(epochs, accs_mean - accs_std, accs_mean + accs_std, alpha=0.3)
plt.title("Accuracy validation"); plt.grid(True);

plt.show();
