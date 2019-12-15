import os
from tqdm import tqdm
import pickle
import numpy as np
import pdb


class Input_feature(object):
    """docstring for Input_feature"""

    def __init__(self, feature, label, idx, seq_len):
        super(Input_feature, self).__init__()
        self.feature = feature
        self.label = label
        self.idx = idx
        self.seq_len = seq_len


def preprocess():
    with open('./data/annotation_test.pkl', 'rb') as f:
        annotation_test = pickle.load(f, encoding='iso-8859-1')
    with open('./data/annotation_training.pkl', 'rb') as f:
        annotation_training = pickle.load(f, encoding='iso-8859-1')
    with open('./data/annotation_validation.pkl', 'rb') as f:
        annotation_validation = pickle


if __name__ == '__main__':
    preprocess()
