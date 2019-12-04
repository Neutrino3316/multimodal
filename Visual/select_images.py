import os
import random
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np

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


def randomGetImage(all_image, interval):
    length = len(all_image)
    print('length: ', length)
    images = []
    if length <= interval:
        for i in range(1, length + 1):
            name = 'frame' + str(i)
            images.append(name)
        return images
    sep = length // interval
    for i in range(interval):
        final = (i + 1) * sep
        if final > length:
            final = length
        else:
            pass
        get_frame = random.randint(sep * i + 1, final)
        name = 'frame' + str(get_frame)
        images.append(name)
    return images


all_images = {}
path = 'data/ImageData/trainingData'
files = os.listdir(path)
for file in files:
    images_path = os.path.join(path, file)
    all_image = os.listdir(images_path)
    interval = 0
    if np.mean(videos_OCEAN_labels[file + '.mp4']) < 0.5 or np.mean(videos_OCEAN_labels[file + '.mp4']) > 0.7:
        interval = 30
    else:
        interval = 15
    all_images[file] = randomGetImage(all_image, interval)
    try:
        if not os.path.exists('data/ImageData/select_trainingData/' + file):
            os.makedirs('data/ImageData/select_trainingData/' + file)
            for image_name in all_images[file]:
                image = cv2.imread(path + '/' + file + '/' + image_name + '.jpg')
                address = 'data/ImageData/select_trainingData/' + file + '/' + image_name + '.jpg'
                cv2.imwrite(address, image)
            print(path + '/' + file)
    except OSError:
        print('Error: Creating directory of data')
