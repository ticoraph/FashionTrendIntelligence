import os
import numpy as np
import cv2

mask_true = "Mask/"
mask_pred = "MaskPrediction/"
backup_folder_concatenated = "IMGConcat/"
backup_folder_overlay = "IMGOverlay/"

# 2. Définition d'un colormap personnalisé et lecture des paires
# Colormap personnalisé pour 4 classes (valeurs 1 à 4) en BGR
custom_colormap = {
    1: (0, 255, 255),  # Jaune - Hat
    2: (0, 165, 255),  # Orange - Hair
    3: (255, 0, 255),  # Magenta - Sunglasses
    4: (0, 0, 255),  # Rouge - Upper-clothes
    5: (255, 255, 0),  # Cyan - Skirt
    6: (0, 255, 0),  # Vert - Pants
    7: (255, 0, 0),  # Bleu - Dress
    8: (128, 0, 128),  # Violet - Belt
    9: (0, 255, 255),  # Jaune - Left-shoe
    10: (255, 140, 0),  # Orange foncé - Right-shoe
    11: (200, 180, 140),  # Beige - Face
    12: (200, 180, 140),  # Beige - Left-leg
    13: (200, 180, 140),  # Beige - Right-leg
    14: (200, 180, 140),  # Beige - Left-arm
    15: (200, 180, 140),  # Beige - Right-arm
    16: (0, 128, 255),  # Bleu clair - Bag
    17: (255, 20, 147)  # Rose - Scarf
}

# Légendes associées aux labels
legend_labels = {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    "3": "Sunglasses",
    "4": "Upper-clothes",
    "5": "Skirt",
    "6": "Pants",
    "7": "Dress",
    "8": "Belt",
    "9": "Left-shoe",
    "10": "Right-shoe",
    "11": "Face",
    "12": "Left-leg",
    "13": "Right-leg",
    "14": "Left-arm",
    "15": "Right-arm",
    "16": "Bag",
    "17": "Scarf"
}

# Lecture des paires image/mask depuis les dossiers
image_files = sorted(os.listdir(mask_true))
mask_files = sorted(os.listdir(mask_pred))
# image_files = sorted([str(x) for x in os.listdir(mask_true)])
# mask_files = sorted([str(x) for x in os.listdir(mask_pred)])

paires = []
for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(mask_true, img_file)
    mask_path = os.path.join(mask_pred, mask_file)

    image = cv2.imread(img_path)  # Image en couleur
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Masque en niveaux de gris
    idx = (img_file.split("_")[1]).split(".")[0]

    new_data = {
        'image': image,
        'mask': mask,
        'idx': idx,
    }
    paires.append(new_data)

print(paires)
print("Lecture des paires image/mask effectuée.")


# 3. Fonctions pour coloriser le masque et ajouter la légende

def colorize_mask(mask, colormap):
    """
    Applique le colormap personnalisé au masque.
    Pour chaque pixel, s'il correspond à un label défini dans colormap,
    la couleur correspondante est assignée.
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        colored_mask[mask == label] = color
    return colored_mask


def add_legend(image, legend, start_x=10, start_y=10, box_size=15, spacing=5):
    """
    Ajoute une légende sur l'image.
    Pour chaque label, dessine un rectangle de la couleur correspondante et le texte associé.
    """
    img_with_legend = image.copy()
    y = start_y
    for label, text in legend.items():
        # Récupération de la couleur du label
        color = custom_colormap.get(int(label), (255, 255, 255))
        # Dessin d'un petit rectangle rempli
        cv2.rectangle(img_with_legend, (start_x, y), (start_x + box_size, y + box_size), color, -1)
        # Ajout du texte à droite du rectangle
        cv2.putText(img_with_legend, text, (start_x + box_size + spacing, y + box_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += box_size + spacing
    return img_with_legend


# 4. Application du colormap, ajout de la légende et superposition image/mask

for pair in paires:
    img = pair['image']
    msk = pair['mask']
    idx = pair['idx']

    # Colorisation du masque avec le colormap personnalisé
    colored_mask = colorize_mask(msk, custom_colormap)

    # Ajout de la légende sur le masque colorisé
    colored_mask_with_legend = add_legend(colored_mask, legend_labels)

    # Superposition du masque coloré sur l'image originale
    overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
    overlay_with_legend = add_legend(overlay, legend_labels)

    # Concatenation des images sur une seule ligne
    concatenated = np.hstack([img, colored_mask_with_legend, overlay_with_legend])

    # Affichage des résultats dans Colab
    print(f"Résultat pour la paire {idx} :")

    cv2.imwrite(f"{backup_folder_overlay}/overlay_{idx}.png", overlay_with_legend)
    cv2.imwrite(f"{backup_folder_concatenated}/concat_{idx}.png", concatenated)

    # plt.figure(figsize=(15, 5))
    # plt.imshow(cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title(f"Paire {idx}")
    # plt.show()
