import os
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import torch

import pdb

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
        video_out = []
        for image_name in all_image:
            image = os.path.join(images_path, image_name)
            image = Image.open(image)
            pdb.set_trace()
            video_out.append(image)
        video_out = torch.stack(video_out)

        data[file] = {'feature':video_out}  
    return data  


if __name__ == '__main__':
    dataset = preprocess_image('test')
    pdb.set_trace()