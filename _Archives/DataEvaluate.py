import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns

true_path = "../Mask/"
pred_path = "Segm/"

# ------------------------
# 1. Fonctions métriques
# ------------------------
def pixel_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.size

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 1.0

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 1.0

# ------------------------
# 1. Fonctions métriques
# ------------------------

list_pred_img = sorted(glob.glob(pred_path + '*.png'))
# print(list_pred_img)

list_true_img = sorted(glob.glob(true_path + '/*.png'))
# print(list_true_img)

results = []

for pred_img, true_img in zip(list_pred_img, list_true_img):
    # print(pred_img)
    # print(true_img)

    true_img_open = Image.open(os.path.join(true_img))
    pred_img_open = Image.open(os.path.join(pred_img))

    true_img_convert = np.array(true_img_open.convert("L"))  # niveaux de gris
    true_img_convert = (true_img_convert > 127).astype(np.uint8)  # binarisation 0/1

    pred_img_convert = np.array(pred_img_open.convert("L"))  # niveaux de gris
    pred_img_convert = (pred_img_convert > 127).astype(np.uint8)  # binarisation 0/1

    # print(f"true_img_convert:{true_img_convert}")
    # print(f"pred_img_convert:{pred_img_convert}")

    # Calcul métriques
    acc = pixel_accuracy(true_img_convert, pred_img_convert)
    iou = iou_score(true_img_convert, pred_img_convert)
    dice = dice_score(true_img_convert, pred_img_convert)

    results.append((true_img, acc, iou, dice))
    # print(f"Accuracy: {acc:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")

# ------------------------
# 4. Moyennes globales
# ------------------------

df = pd.DataFrame(results, columns=["filename", "accuracy", "iou", "dice"])
# print(df)

if results:
    acc_mean = np.mean([r[1] for r in results])
    iou_mean = np.mean([r[2] for r in results])
    dice_mean = np.mean([r[3] for r in results])

    print("\n📊 Résultats globaux :")
    print(f"Accuracy moyenne: {acc_mean * 100:.2f}%")
    # Cela veut dire que 77% des pixels de tes images sont correctement classés.
    print(f"IoU moyenne: {iou_mean * 100:.2f}%")
    # en moyenne, la zone prédite et la vérité terrain se chevauchent à 61%
    print(f"Dice moyenne: {dice_mean * 100:.2f}%")
    # le chevauchement pondéré est de 74%
else:
    print("⚠️ Aucun résultat calculé.")

print(df.sort_values("accuracy", ascending=True)[:3])
print(df.sort_values("iou", ascending=True)[:3])

print(df.sort_values("accuracy", ascending=True)[-3:])
print(df.sort_values("iou", ascending=True)[-3:])

# ------------------------
# 4. GRAPHS
# ------------------------
'''
x = np.arange(len(df['filename']))
plt.scatter(x, df['accuracy'], color="blue", label="Accuracy", alpha=0.7)
plt.scatter(x, df['iou'], color="red", label="iou", alpha=0.7)
plt.xlabel("images")
plt.ylabel("score")
# Accuracy vs IoU : intéressant car l’accuracy peut être trompeuse si beaucoup de fond, tu verras si c’est bien corrélé ou pas.
plt.show()

plt.scatter(df['iou'], df['dice'], color="magenta", alpha=0.7)
plt.xlabel("iou")
plt.ylabel("dice")
plt.title("Corrélation iou vs dice")
plt.show()

sns.boxplot(data=df[["accuracy", "iou", "dice"]])
plt.title("Distribution des scores")
plt.show()
'''