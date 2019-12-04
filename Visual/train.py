#################################################################                            ResNet                           ###############################################################
# import os.path

# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from torch.autograd import Variable

# import numpy as np
# from PIL import Image

# features_dir = './data'

# img_path = "data/frame1.jpg"
# file_name = img_path.split('/')[-1]
# feature_path = os.path.join(features_dir, file_name + '.txt')


# transform1 = transforms.Compose([
#     transforms.ToTensor()]
# )

# img = Image.open(img_path)
# img1 = transform1(img)

# #resnet18 = models.resnet18(pretrained = True)
# resnet50_feature_extractor = models.resnet50(pretrained=True)
# resnet50_feature_extractor.fc = nn.Linear(2048, 1000)
# torch.nn.init.eye(resnet50_feature_extractor.fc.weight)

# for param in resnet50_feature_extractor.parameters():
#     param.requires_grad = False
# #resnet152 = models.resnet152(pretrained = True)
# #densenet201 = models.densenet201(pretrained = True)
# x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
# #y1 = resnet18(x)
# y = resnet50_feature_extractor(x)
# y = y.data.numpy()
# np.savetxt(feature_path, y, delimiter=',')
# #y3 = resnet152(x)
# #y4 = densenet201(x)

# y_ = np.loadtxt(feature_path, delimiter=',').reshape(1, 128)
# print(y_)
# print(y_.shape)
############################################################# VGG_16 ##############################################################################################################################################
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
from net import FullConnect, MyBiLSTM
from tqdm import tqdm
from Vgg_face import vgg_face_dag
"""
get labels
the form is {video_name: labels[5]}
"""


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(threshold=1000000000000000)
path = './data/annotation_training.pkl'
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
"""
deal the picture with vgg-16
"""
device = torch.device("cuda")
print('use device: %s' % device, file=sys.stderr)
vgg = vgg_face_dag('./data/vgg_face_dag.pth')
# vgg = vgg.to(device)
# pre_data = torch.load('./data/vgg16-397923af.pth')
# vgg.load_state_dict(pre_data)
print("finish load vgg model")

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

losses = []
loss_function = nn.MSELoss()
model = MyBiLSTM(2622, 2622, 1, 5, vgg)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
all_images = {}
path = 'data/ImageData/select_trainingData'
files = os.listdir(path)
print("finish get path")
print("start training")
i = 0

small_test = 0  # use for small sample test
for epoch in range(20):
    total_loss = 0
    for batch_files, train_labels in batch_iter(files, videos_OCEAN_labels, batch_size=256):
        print('start new batch')
        count_file = 0
        for file in tqdm(batch_files):
            count_file += 1
            images_path = os.path.join(path, file)
            all_image = os.listdir(images_path)
            all_images[file] = all_image
            log_probs = torch.zeros([1, 5])
            label = train_labels[file + '.mp4']
            label = torch.FloatTensor(label)
            label = torch.unsqueeze(label, 0)
            label = label.to(device)
            video_out = torch.zeros([1, 3, 224, 224])
            for image_name in all_images[file]:
                image = path + '/' + file + '/' + image_name    # os.path.join()
                image = Image.open(image)
                image = trans(image)
                image.unsqueeze_(dim=0)
                # print(image.shape)
                # image = image.to(device)
                # out = vgg(image)
                video_out = torch.cat((video_out, image), 0)
            video_out = video_out[1:, :]
            # video_out = video_out.unsqueeze(1)
            model.zero_grad()
            log_prob = model(video_out.to(device))
            loss = loss_function(log_prob, label)
            loss.backward()
            optimizer.step()
            total_loss += loss
    print('finish epoch: ', epoch)
    losses.append(total_loss)
    torch.save(model, './model5/model' + str(epoch) + '.pkl')
print(losses)
