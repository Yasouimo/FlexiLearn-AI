def resize_and_normalize(image, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

def load_image_from_zip(zip_file_buffer, file_path):
    with zipfile.ZipFile(zip_file_buffer, 'r') as zf:
        with zf.open(file_path) as f:
            image = Image.open(f).convert('RGB')
    return image

def create_image_dataset(zip_file_buffer, transform=None, root_folder=''):
    image_paths = []
    labels = []
    class_to_idx = {}
    
    if root_folder and not root_folder.endswith('/'):
        root_folder += '/'
    
    with zipfile.ZipFile(zip_file_buffer, 'r') as zf:
        class_names = set()
        for file_path in zf.namelist():
            if file_path.startswith(root_folder) and not file_path.startswith('__MACOSX'):
                relative_path = file_path[len(root_folder):]
                if '/' in relative_path:
                    class_names.add(relative_path.split('/')[0])
        
        class_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}
        
        for file_path in zf.namelist():
            if file_path.endswith(('.png', '.jpg', '.jpeg')) and file_path.startswith(root_folder) and not file_path.startswith('__MACOSX'):
                relative_path = file_path[len(root_folder):]
                if '/' in relative_path:
                    class_name = relative_path.split('/')[0]
                    if class_name in class_to_idx:
                        image_paths.append(file_path)
                        labels.append(class_to_idx[class_name])
    
    return image_paths, labels, class_to_idx