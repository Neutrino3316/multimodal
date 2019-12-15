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
"""
get labels
the form is {video_name: labels[5]}
"""


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda: 0")
print('use device: %s' % device, file=sys.stderr)
np.set_printoptions(threshold=1000000000000000)
path = './data/annotation_validation.pkl'
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

collect_all_log = []
for epoch in range(1):
    vgg_model = torch.load('./model3/vgg_model' + str(4) + '.pkl')
    LSTM_model = torch.load('./model3/LSTM_model' + str(4) + '.pkl')
    vgg_model = vgg_model.to(device)
    model_state_dict = LSTM_model.state_dict()
    FC_new_model = MyBiLSTM(1000, 1000, 1, 5)
    FC_new_model.load_state_dict(model_state_dict)
    FC_new_model.to(device)

    losses = []
    all_images = {}
    path = 'data/ImageData/common_validation'
    files = os.listdir(path)
    print("finish get path")
    print("start validation")
    with torch.no_grad():
        loss = torch.zeros([1, 5])
        log_probs = torch.zeros([1, 5])
        for batch_files, train_labels in batch_iter(files, videos_OCEAN_labels, batch_size=256):
            print('start new batch')
            for file in batch_files:
                images_path = os.path.join(path, file)
                all_image = os.listdir(images_path)
                all_images[file] = all_image
                label = train_labels[file + '.mp4']
                label = torch.FloatTensor(label)
                label = torch.unsqueeze(label, 0)
                video_out = torch.zeros([1, 1000])
                for image_name in all_images[file]:
                    image = path + '/' + file + '/' + image_name
                    image = Image.open(image)
                    image = trans(image)
                    image.unsqueeze_(dim=0)
                    image = image.to(device)
                    out = vgg_model(image)
                    video_out = torch.cat((video_out, out.cpu()), 0)
                video_out = video_out[1:, :]
                video_out = video_out.unsqueeze(1)
                log_prob = FC_new_model(video_out.to(device))
                log_probs += log_prob.cpu()
                collect_all_log.append(log_prob)
                loss += (torch.abs(log_prob.cpu() - label.cpu()))
        print(1 - loss / len(files))
        print(log_probs / len(files))
        np.savetxt('./data/collect_all_log.txt', collect_all_log)
