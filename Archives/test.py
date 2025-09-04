import time
from dotenv import load_dotenv
import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import base64
import io
import evaluate
import torch
import torch.nn as nn

image_path_full = "../IMG/image_0.png"

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

print(detect_content_type(image_path_full))



'''
# Function to get content type
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
    '''
