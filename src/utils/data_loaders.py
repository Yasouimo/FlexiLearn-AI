import io
import zipfile
from PIL import Image
import pandas as pd

def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

def load_excel(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {e}")

def load_zip_images(zip_file, transform=None, root_folder=''):
    from src.models.custom_datasets import ImageZipDataset
    zip_buffer = io.BytesIO(zip_file.getvalue())
    return ImageZipDataset(zip_buffer, transform=transform, root_folder=root_folder)