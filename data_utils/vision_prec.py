import os
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import torch


path = '../dataset/raw_data/vision/'
def preprocess_image(data_type):
    files_dir = os.path.join(path, data_type)
    files_name = os.listdir(files_dir)
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    data = {}
    for file in files_name:
        images_path = os.path.join(files_dir, file)
        all_image = os.listdir(images_path)
        video_out = torch.zeros([1, 3, 224, 224])
        for image_name in all_image:
            image = os.path.join(images_path, image_name)
            image = Image.open(image)
            image = trans(image)
            image.unsqueeze_(dim=0)
            video_out = torch.cat((video_out, image), 0)
        video_out = video_out[1:, :]
        data[file] = {'feature':video_out}  
    return data  
