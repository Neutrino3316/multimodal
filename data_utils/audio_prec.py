import os
import pickle

import numpy as np

from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import ShortTermFeatures


path = "../dataset/raw_data/audio/"
def preprocess_audio(data_type):
    files_dir = os.path.join(path, data_type)
    files_name = os.listdir(files_dir)
    mp3_files = filter(lambda file: file.split(".")[-1] == "mp3", files_name)  # filter out files in mp3 format
    mp3_files = list(mp3_files)

    data = dict()
    for file in mp3_files:
        [Fs, x] = audioBasicIO.read_audio_file(os.path.join(path, file))
        try:
            F0, _ = ShortTermFeatures.feature_extraction(x[:, 0], Fs, 0.050 * Fs, 0.025 * Fs)
            F1, _ = ShortTermFeatures.feature_extraction(x[:, 1], Fs, 0.050 * Fs, 0.025 * Fs)
        except IndexError:
            F0, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
            F1 = np.zeros(F0.shape)
        feature = np.concatenate([F0, F1], axis=0)

        seq_len = feature.shape[1]
        if seq_len < 611:   # if seq_len < 611, pad to 611
            new_feature = np.zeros((68, 611))
            new_feature[:, :seq_len] = feature
            feature = new_feature

        utterance_id = file[:-4]
        data[utterance_id] = {'feature': feature, 'seq_len': seq_len}
        
    return data


if __name__ == '__main__':
    dataset = preprocess_audio('train')