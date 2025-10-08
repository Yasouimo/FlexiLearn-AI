# File: /Streamlit_app_deep/Streamlit_app_deep/src/models/custom_datasets.py

import zipfile
from PIL import Image
from torch.utils.data import Dataset

class ImageZipDataset(Dataset):
    """Custom dataset for loading images from ZIP file"""
    def __init__(self, zip_file_buffer, transform=None, root_folder=''):
        self.zip_file_buffer = zip_file_buffer
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        if root_folder and not root_folder.endswith('/'):
            root_folder += '/'
        
        with zipfile.ZipFile(self.zip_file_buffer, 'r') as zf:
            class_names = set()
            for file_path in zf.namelist():
                if file_path.startswith(root_folder) and not file_path.startswith('__MACOSX'):
                    relative_path = file_path[len(root_folder):]
                    if '/' in relative_path:
                        class_names.add(relative_path.split('/')[0])
            
            self.class_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}
            self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
            
            for file_path in zf.namelist():
                if file_path.endswith(('.png', '.jpg', '.jpeg')) and file_path.startswith(root_folder) and not file_path.startswith('__MACOSX'):
                    relative_path = file_path[len(root_folder):]
                    if '/' in relative_path:
                        class_name = relative_path.split('/')[0]
                        if class_name in self.class_to_idx:
                            self.image_paths.append(file_path)
                            self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_file_buffer, 'r') as zf:
            with zf.open(self.image_paths[idx]) as f:
                image = Image.open(f).convert('RGB')
        
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label