'''
lables name:
extraversion
neuroticism
agreeableness
conscientiousness
interview
openness
'''
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import sys
sys.getdefaultencoding()
import pickle
import numpy as np
import zipfile
import os
import random
from utils import batch_iter
import matplotlib.pyplot as plt
from net import FullConnect, MyBiLSTM
from Vgg_face import vgg_face_dag
"""
get labels
the form is {video_name: labels[5]}
"""


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
print('use device: %s' % device, file=sys.stderr)
np.set_printoptions(threshold=1000000000000000)
path = './data/annotation_test.pkl'
file = open(path, 'rb')
labels = pickle.load(file, encoding='iso-8859-1')  # 读取pkl文件的内容
videos_name = []
videos_OCEAN_labels = {}
for key in labels['openness']:
    videos_name.append(key)
for name in videos_name:
    label = []
    label.append(labels['openness'][name])
    label.append(labels['conscientiousness'][name])
    label.append(labels['extraversion'][name])
    label.append(labels['agreeableness'][name])
    label.append(labels['neuroticism'][name])
    videos_OCEAN_labels[name] = label

print("finish read labels")
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

vgg = vgg_face_dag()
FC_model = torch.load('./model4/model4.pkl')
model_state_dict = FC_model.state_dict()
FC_new_model = MyBiLSTM(1000, 1000, 1, 5, vgg)
FC_new_model.load_state_dict(model_state_dict)
FC_new_model.to(device)

losses = []
loss_function = nn.MSELoss()
all_images = {}
path = 'data/ImageData/select_testData'
files = os.listdir(path)
print("finish get path")
print("start validation")
i = 0
with torch.no_grad():
    loss = torch.zeros([1, 5])
    for batch_files, test_labels in batch_iter(files, videos_OCEAN_labels, batch_size=256):
        print('start new batch')
        for file in batch_files:
            images_path = os.path.join(path, file)
            all_image = os.listdir(images_path)
            all_images[file] = all_image
            label = test_labels[file + '.mp4']
            label = torch.FloatTensor(label)
            label = torch.unsqueeze(label, 0)
            video_out = torch.zeros([1, 3, 224, 224])
            for image_name in all_images[file]:
                image = path + '/' + file + '/' + image_name
                image = Image.open(image)
                image = trans(image)
                image.unsqueeze_(dim=0)
                # image = image.to(device)
                # out = vgg_model(image)
                video_out = torch.cat((video_out, image), 0)

            video_out = video_out[1:, :]
            # video_out = video_out.unsqueeze(1)
            log_prob = FC_new_model(video_out.to(device))
            loss += torch.abs(log_prob.cpu() - label.cpu())
    print(1 - loss / len(files))
