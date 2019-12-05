from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import ShortTermFeatures
import os
from tqdm import tqdm
import pickle
import numpy as np
import pdb
# import matplotlib.pyplot as plt

class Input_feature():
    def __init__(self, feature, label, seq_len, unique_idx):
        self.feature = feature
        self.label = label
        self.seq_len = seq_len
        self.unique_idx = unique_idx


def preprocess():
    # read in annotations
    with open("../dataset/annotations/annotation_test.pkl", "rb") as f:
        annotations_test = pickle.load(f, encoding='iso-8859-1')

    with open("../dataset/annotations/annotation_training.pkl", "rb") as f:
        annotations_train = pickle.load(f, encoding='iso-8859-1')

    with open("../dataset/annotations/annotation_validation.pkl", "rb") as f:
        annotations_valid = pickle.load(f, encoding='iso-8859-1')

    # read in audio files, and preprocess for feature extraction
    outputs = []
    paths = ["../dataset/trainset/", "../dataset/validset/", "../dataset/testset/"]
    sets = [('train', annotations_train), ('valid', annotations_valid), ('test', annotations_test)]
    label_types = annotations_test.keys()   # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    idx2name = dict()
    unique_idx = 1
    for s, annotations in sets:
        path = os.path.join("../dataset/", s + "set")
        dir = os.listdir(path)
        mp3_files = filter(lambda file: file.split(".")[-1] == "mp3", dir)  # filter out files in mp3 format
        mp3_files = list(mp3_files)

        dataset = []
        # features, labels = [], []
        count = 0
        for file in tqdm(mp3_files):
            [Fs, x] = audioBasicIO.read_audio_file(os.path.join(path, file))
            try:
                F0, _ = ShortTermFeatures.feature_extraction(x[:, 0], Fs, 0.050 * Fs, 0.025 * Fs)
                F1, _ = ShortTermFeatures.feature_extraction(x[:, 1], Fs, 0.050 * Fs, 0.025 * Fs)
            except IndexError:
                F0, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
                F1 = np.zeros(F0.shape)
                count += 1
            feature = np.concatenate([F0, F1], axis=0)

            seq_len = feature.shape[1]
            if seq_len < 611:   # if seq_len < 611, pad to 611
                new_feature = np.zeros((68, 611))
                new_feature[:, :seq_len] = feature
                feature = new_feature

            key = file[:-1] + '4'
            idx2name[unique_idx] = key
            label = np.array([annotations[tp][key] for tp in label_types])
            # labels.append(label)
            dataset.append(Input_feature(feature, label, seq_len, unique_idx))
            unique_idx += 1

        # dataset = (features, labels)

        filename = os.path.join("../dataset/preprocessed/", "preproc_audio_" + s + ".pkl")
        with open(filename, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Total files of {s}set: {len(dataset)}, files of single audio stream: {count}")
        outputs.append(dataset)

    filename = os.path.join("../dataset/preprocessed/", "idx_dict.pkl")
    with open(filename, "wb") as f:
        pickle.dump(idx2name, f)

    outputs.append(idx2name)
    return outputs


# path = "../dataset/validset/"
# dir = os.listdir(path)
# mp3_files = filter(lambda file: file.split(".")[-1] == "mp3", dir)  # filter out files in mp3 format
# mp3_files = list(mp3_files)
# count = 0
# for file in tqdm(mp3_files):
#     [Fs, x] = audioBasicIO.readAudioFile(os.path.join(path, file))
#
#     try:
#         F0, _ = audioFeatureExtraction.stFeatureExtraction(x[:, 0], Fs, 0.050 * Fs, 0.025 * Fs)
#         F1, _ = audioFeatureExtraction.stFeatureExtraction(x[:, 1], Fs, 0.050 * Fs, 0.025 * Fs)
#         F = np.concatenate([F0, F1], axis=0)
#     except IndexError:
#         F, _ = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#         print(x.shape)
#         count += 1
# print(count)