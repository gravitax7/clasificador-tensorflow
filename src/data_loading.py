import tensorflow as tf
import os
from PIL import Image
import numpy as np

class ImageDatasetLoader:
    def __init__(self, data_dir, img_size=(128, 128)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = os.listdir(data_dir)  # Asume subcarpetas por clase

    def load_data(self):
        images, labels = [], []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path).convert("RGB").resize(self.img_size)
                img_array = np.array(img) / 255.0  # Normalización
                images.append(img_array)
                labels.append(class_idx)
        return np.array(images), np.array(labels)