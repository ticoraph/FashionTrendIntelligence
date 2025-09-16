import glob
import os
import numpy as np
import torch
from PIL import Image

target_path = "../IMG/"
pred_path = "../IMGOverlay/"

# Function to evaluate metrics
def compute(pred, target, num_classes):
    """
    Calcule IoU, Dice Score et Accuracy par classe et en moyenne.
    pred : masque prédit (H, W)
    target : masque vérité terrain (H, W)
    num_classes : nombre total de classes
    """
    ious, dices, accuracies = [], [], []

    # Calcul de l'accuracy globale (pixel-wise)
    total_pixels = pred.numel()
    correct_pixels = (pred == target).sum().item()
    global_accuracy = correct_pixels / total_pixels

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()
        union = pred_sum + target_sum - intersection

        # IoU
        iou = intersection / union if union > 0 else float("nan")
        ious.append(iou)

        # Dice
        dice = (2 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else float("nan")
        dices.append(dice)

        # Accuracy par classe (True Positive Rate / Recall / Sensitivity)
        accuracy = intersection / target_sum if target_sum > 0 else float("nan")
        accuracies.append(accuracy)

    # Moyennes (en ignorant les NaN)
    mean_iou = torch.tensor([x for x in ious if not torch.isnan(torch.tensor(x))]).mean().item()
    mean_dice = torch.tensor([x for x in dices if not torch.isnan(torch.tensor(x))]).mean().item()
    mean_accuracy = torch.tensor([x for x in accuracies if not torch.isnan(torch.tensor(x))]).mean().item()

    return ious, dices, accuracies, mean_iou, mean_dice, mean_accuracy, global_accuracy


# --- Exemple d’utilisation ---
# Simule une sortie prédite et un masque vérité terrain avec 3 classes
# Charger les images
#list_pred_img = sorted(os.listdir(pred_path))
list_pred_img = sorted(glob.glob(pred_path + '*.png'))
print(list_pred_img)
#list_target_img = sorted(os.listdir(target_path))
list_target_img = sorted(glob.glob(target_path + '/*.png'))
print(list_target_img)

for pred_img,target_img in zip(list_pred_img, list_target_img):
    print(pred_img)
    print(target_img)
    pred_img = Image.open(os.path.join(pred_img))
    target_img = Image.open(os.path.join(target_img))

    # Convertir en tenseurs
    pred = torch.tensor(np.array(pred_img), dtype=torch.long)
    target = torch.tensor(np.array(target_img), dtype=torch.long)

    ious = compute(pred, target, num_classes=18)
    mean_iou = compute(pred, target, num_classes=18)
    dices = compute(pred, target, num_classes=18)
    mean_dice = compute(pred, target, num_classes=18)
    accuracies = compute(pred, target, num_classes=18)
    mean_accuracy = compute(pred, target, num_classes=18)
    global_accuracy = compute(pred, target, num_classes=18)

    #print("IoU par classe:", ious)
    print("Mean IoU:", mean_iou)
    #print("Dice par classe:", dices)
    print("Mean Dice:", mean_dice)
    #print("Accuracies:", accuracies)
    print("Mean Accuracy:", mean_accuracy)
    #print("Global Accuracy:", global_accuracy)