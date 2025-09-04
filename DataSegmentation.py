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
image_dir = "IMG/"
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
    # Open the image file in binary read mode
    with open(image_path, "rb") as image_file:
        # Read the file content, encode it to base64, and convert to UTF-8 string
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Return the base64 string with proper data URI format for JPEG images
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
    # Decode the base64 string to binary data
    mask_data = base64.b64decode(base64_string)
    # Create an image from the binary data
    mask_image = Image.open(io.BytesIO(mask_data))
    # Convert the image to a NumPy array
    mask_array = np.array(mask_image)
    # If the mask has multiple channels (RGB), take only the first channel
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    # Resize the mask to the target dimensions using nearest neighbor interpolation
    # to preserve the binary/categorical nature of the mask
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    # Convert back to NumPy array and return
    return np.array(mask_image)


# %%
def get_image_dimensions(image_path):
    """
    Get the dimensions of an image.
    Args:
        image_path (str): Path to the image.
    Returns:
        tuple: (width, height) of the image.
    """
    # Open the image file from the specified path
    original_image = Image.open(image_path)
    # Return the image dimensions as a tuple (width, height)
    return original_image.size


# %%
def detect_content_type(image_path_full):
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
    except Exception:
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
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0  # Class ID for Background is 0

    return combined_mask


# %%
def query_huggingface(model_name, image_path, api_token):
    """
    Request API Hugging Face.
    Args:
        model_name: Nom du Modele
        image_path: Image
        api_token: Token
    Returns:
        response.json: Reponse Requete API
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    base64_image = encode_image_to_base64(image_path)
    payload = {"inputs": base64_image}

    try:

        session = requests.Session()
        retries = Retry(total=3)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        #response = requests.post(api_url, headers=headers, json=payload, timeout=5)
        response = session.post(api_url, headers=headers, json=payload)
        print(response.json())

        response.raise_for_status()
        # between 200–299 (successful responses), the method does nothing.
        # 4xx (client error) or 5xx (server error), it raises an HTTPError with details of the failed request.
        print(f" ✅ Success! Code de statut: {response.status_code}")
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")

    except Exception as e:
        print(f"Error during API request {image_path}: {e}")
        return None


# %%
def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.
    Args:
        list_of_image_paths (list): Liste des chemins vers les images.
    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []

    for image_path in tqdm(list_of_image_paths,
                           desc="Segmentation",
                           unit="image",
                           colour="green"):
        try:
            image_path_full = image_dir + image_path
            # print(f"Traitement: {image_path}")

            # Image Dimensions
            image_dimensions = get_image_dimensions(image_path_full)
            #print(f"Dimensions: {image_dimensions}")

            # Content Type
            content_type = detect_content_type(image_path_full)
            #print(f"Content Type: {content_type}")

            if content_type is "image/png" and image_dimensions == (400, 600):

                # API request
                # mattmdjaga/segformer_b2_clothes
                # sayeed99/segformer_b3_clothes
                result = query_huggingface("sayeed99/segformer_b3_clothes", image_path_full, api_token)
                # Create Mask
                combined_mask = create_masks(result, image_dimensions[0], image_dimensions[1])
                # Append List
                batch_segmentations.append(combined_mask)
            else:
                batch_segmentations.append(None)
        except Exception as e:
            print(f"Erreur lors du traitement de {image_path}: {e}")

            batch_segmentations.append(None)

    return batch_segmentations


# %%
def display_segmented_images_batch(list_of_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.
    Args:
        list_of_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
    """

    for image_path, segmentation_mask in zip(list_of_image_paths, segmentation_masks):
        # print(image_path)
        # print(segmentation_mask)
        image_open = Image.open(image_dir + image_path)

        # Premier sous-graphique
        plt.subplot(1, 2, 1)
        plt.imshow(image_open)
        plt.title('Image Original')
        plt.axis('off')

        # Deuxième sous-graphique
        plt.subplot(1, 2, 2)
        plt.imshow(segmentation_mask, cmap='tab10', interpolation='nearest')
        plt.title('Masque de segmentation')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Si le masque possède 3 canaux, le convertir en niveaux de gris
        # if segmentation_mask.ndim == 3:
        #    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        # id = (image_path.split("_")[1]).split(".")[0]
        # Sauvegarde du masque dans les dossiers appropriés
        # cv2.imwrite(os.path.join("MaskFromScript", f"mask_{id}.png"), segmentation_mask)


# %%
# Appeler la fonction pour segmenter les images listées dans image_paths
if list_of_image_paths:
    print(f"\nTraitement de {len(list_of_image_paths)} image(s) en batch...")
    batch_seg_results = segment_images_batch(list_of_image_paths)
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")

# Appeler la fonction pour afficher les images originales et leurs segmentations
if batch_seg_results:
    display_segmented_images_batch(list_of_image_paths, batch_seg_results)
else:
    print("Aucun résultat de segmentation à afficher.")