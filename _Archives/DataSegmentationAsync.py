# %%
from dotenv import load_dotenv
import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import base64
import io
import aiohttp
import asyncio
import time
from urllib3 import Retry

# %%
load_dotenv()
api_token = os.getenv('FASHION_TREND_INTELLIGENCE_TOKEN_READ')
image_dir = "../IMG/"
list_of_image_paths = os.listdir(image_dir)

# %%
CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}


# %%
def encode_image_to_base64(image_path):
    """Encode une image en base64."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"


# %%
def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.
    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.
    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)

    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


# %%
def get_image_dimensions(image_path):
    """Get the dimensions of an image."""
    original_image = Image.open(image_path)
    return original_image.size


# %%
def detect_content_type(image_path_full):
    """Détecte le type de contenu d'une image."""
    try:
        with Image.open(image_path_full) as img:
            format_to_mime = {
                'JPEG': 'image/jpeg',
                'PNG': 'image/png',
                'GIF': 'image/gif',
                'BMP': 'image/bmp',
                'WEBP': 'image/webp',
                'TIFF': 'image/tiff',
                'ICO': 'image/x-icon',
            }
            return format_to_mime.get(img.format, f'image/{img.format.lower()}')
    except Exception as e:
        print(f"Error {e}")
        return None


# %%
def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.
    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.
    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            combined_mask[mask_array > 0] = 0

    return combined_mask


# %%
async def query_huggingface_async(session, model_name, image_path, token, semaphore):
    """
    Requête API Hugging Face asynchrone avec limitation de concurrence.
    Args:
        session: Session aiohttp
        model_name: Nom du modèle
        image_path: Chemin vers l'image
        token: Token API
        semaphore: Semaphore pour limiter la concurrence
    Returns:
        dict: Réponse de l'API ou None en cas d'erreur
    """
    async with semaphore:  # Limite le nombre de requêtes simultanées
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        try:
            base64_image = encode_image_to_base64(image_path)
            payload = {"inputs": base64_image}

            async with session.post(api_url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f" ✅ Success! Image: {os.path.basename(image_path)}, Status: {response.status}")
                    return result
                else:
                    print(f" ❌ Error! Image: {os.path.basename(image_path)}, Status: {response.status}")
                    return None

        except asyncio.TimeoutError:
            print(f" ⏰ Timeout pour {os.path.basename(image_path)}")
            return None
        except Exception as e:
            print(f" ❌ Erreur pour {os.path.basename(image_path)}: {e}")
            return None


# %%
async def segment_images_batch_async(list_of_image_paths, max_concurrent=5):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face de manière asynchrone.
    Args:
        list_of_image_paths (list): Liste des chemins vers les images.
        max_concurrent (int): Nombre maximum de requêtes simultanées.
    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
    """
    batch_segmentations = []
    start_time = time.time()

    # Création d'un semaphore pour limiter les requêtes simultanées
    semaphore = asyncio.Semaphore(max_concurrent)

    # Création d'une session aiohttp avec timeout configuré
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        valid_images = []

        # Préparer les tâches pour les images valides
        for image_path in list_of_image_paths:
            try:
                image_path_full = os.path.join(image_dir, image_path)
                print(f"Préparation: {image_path}")

                # Vérification des dimensions et du type
                image_dimensions = get_image_dimensions(image_path_full)
                content_type = detect_content_type(image_path_full)

                print(f"Dimensions: {image_dimensions}, Type: {content_type}")

                if content_type == "image/png" and image_dimensions == (400, 600):
                    # Créer la tâche asynchrone
                    task = query_huggingface_async(
                        session,
                        "sayeed99/segformer_b3_clothes",
                        image_path_full,
                        api_token,
                        semaphore
                    )
                    tasks.append(task)
                    valid_images.append((image_path, image_path_full, image_dimensions))
                else:
                    print(f"Image ignorée: {image_path}")

            except Exception as e:
                print(f"Erreur lors de la préparation de {image_path}: {e}")

        if not tasks:
            print("Aucune image valide à traiter.")
            return []

        print(f"\nLancement de {len(tasks)} requêtes asynchrones...")

        # Exécuter toutes les tâches avec une barre de progression
        results = []
        for task in tqdm(asyncio.as_completed(tasks),
                         total=len(tasks),
                         desc="Requêtes API",
                         unit="req",
                         colour="blue"):
            result = await task
            results.append(result)

        # Traiter les résultats dans l'ordre original
        for i, (image_path, image_path_full, image_dimensions) in enumerate(valid_images):
            try:
                if results[i] is not None:
                    # Créer le masque combiné
                    combined_mask = create_masks(results[i], image_dimensions[0], image_dimensions[1])
                    batch_segmentations.append(combined_mask)
                    print(f"✅ Masque créé pour {image_path}")
                else:
                    batch_segmentations.append(None)
                    print(f"❌ Échec pour {image_path}")
            except Exception as e:
                print(f"Erreur lors de la création du masque pour {image_path}: {e}")
                batch_segmentations.append(None)

    elapsed_time = time.time() - start_time
    print(f"\n🕒 Temps total: {elapsed_time:.2f} secondes")
    print(f"📊 Moyenne: {elapsed_time / len(tasks):.2f} sec/image")

    return batch_segmentations


# %%
def display_segmented_images_batch(list_of_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.
    """
    for image_path, segmentation_mask in zip(list_of_image_paths, segmentation_masks):
        if segmentation_mask is None:
            print(f"Pas de masque pour {image_path}")
            continue

        image_open = Image.open(os.path.join(image_dir, image_path))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image_open)
        plt.title('Image Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmentation_mask, cmap='tab10', interpolation='nearest')
        plt.title('Masque de segmentation')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# %%
# Fonction principale pour exécuter le traitement asynchrone
async def main():
    """Fonction principale asynchrone."""
    if list_of_image_paths:
        print(f"\n🚀 Traitement asynchrone de {len(list_of_image_paths)} image(s)...")
        batch_seg_results = await segment_images_batch_async(list_of_image_paths, max_concurrent=3)
        print("✅ Traitement asynchrone terminé.")

# %%
# Exécution du code principal
if __name__ == "__main__":
    # Pour exécuter la version asynchrone
    print("=== VERSION ASYNCHRONE ===")
    asyncio.run(main())

    # Si vous voulez utiliser la version synchrone, décommentez les lignes suivantes:
    # print("\n=== VERSION SYNCHRONE ===")
    # if list_of_image_paths:
    #     batch_seg_results = segment_images_batch(list_of_image_paths)
    #     if batch_seg_results:
    #         display_segmented_images_batch(list_of_image_paths, batch_seg_results)

    """
    # Afficher les résultats
        if batch_seg_results and any(mask is not None for mask in batch_seg_results):
            # Filtrer pour n'afficher que les images avec des masques valides
            valid_results = [(path, mask) for path, mask in zip(list_of_image_paths, batch_seg_results) if
                             mask is not None]
            if valid_results:
                paths, masks = zip(*valid_results)
                display_segmented_images_batch(list(paths), list(masks))
            else:
                print("Aucun résultat valide à afficher.")
        else:
            print("Aucun résultat de segmentation à afficher.")
    else:
        print("Aucune image à traiter.")
        """
