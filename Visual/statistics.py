############################################ get great perform data ######################################################
# import sys
# import os
# path = 'data/ImageData/select_trainingData'
# files = os.listdir(path)
# files_title = [file.split('.')[0] for file in files]
# great_perform = './data/great_perform_file.txt'
# great_files = []
# with open(great_perform, 'r') as file_read:
#     while True:
#         line = file_read.readline()
#         print(line)
#         if not line:
#             break
#             pass
#         split_file = [file for file in line.split()]
#         great_files.append(split_file[1].split('.')[0])
# count = 0
# for file in great_files:
#     if file in files_title:
#         count += 1
# print(count)
####################################### statistic common data and no common data #############################################
# import sys
# import os
# import shutil
# path = 'data/ImageData/select_trainingData'
# select_train = os.listdir(path)
# path2 = 'data/ImageData/select_testData'
# select_validation = os.listdir(path2)
# select_train_names = [file.split('.')[0] for file in select_train]

# for select_dir in select_validation:
# 	select_file = os.path.join(path2, select_dir)
# 	if select_dir.split('.')[0] in select_train_names:
# 		target_file = 'data/ImageData/common_test'
# 	else:
# 		target_file = 'data/ImageData/no_common_test'
# 	target_dir = os.path.join(target_file, select_dir)
# 	if not os.path.exists(target_dir):
# 		os.makedirs(target_dir)
# 	for file in os.listdir(select_file):
# 		source_file = os.path.join(select_file, file)
# 		shutil.copy(source_file, target_dir)
################################### statistic prediction ######################################################################
# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# """
# openness
# """
# openness_label = []
# consc_label = []
# extr_label = []
# agree_label = []
# neuroticism_label = []
# np.set_printoptions(threshold=1000000000000000)
# path = './data/annotation_training.pkl'
# file = open(path, 'rb')
# labels = pickle.load(file, encoding='iso-8859-1')  # 读取pkl文件的内容
# videos_name = []
# videos_OCEAN_labels = {}
# for key in labels['openness']:
#     videos_name.append(key)
# for name in videos_name:
#     label = []
#     label.append(labels['openness'][name])
#     label.append(labels['conscientiousness'][name])
#     label.append(labels['extraversion'][name])
#     label.append(labels['agreeableness'][name])
#     label.append(labels['neuroticism'][name])
#     videos_OCEAN_labels[name] = label
#     openness_label.append(np.mean(label))


# path = 'collect_all_log_test.txt'
# a = np.loadtxt(path)
# # plt.hist(a[:321], density=1, )
# # plt.hist(a[321:], density=1, color='red')
# plt.hist(openness_label, density=1, color='black')

# plt.show()
# # plt.xlim(0.0, 1.0)
################################################################### home work #####################################################################
import torch
from transformers import *
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "[CLS] Who was Jim Henson ? [SEP]"
tokenized_text = tokenizer.tokenize(text)
masked_index = 3
tokenized_text[masked_index] = '[MASK]'
index_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0]
tokens_tensor = torch.tensor([index_tokens])
segments_tensor = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    encoder_layer, output = model(tokens_tensor, segments_tensor)
print(output)
