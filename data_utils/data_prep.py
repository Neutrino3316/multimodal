# This file reads in features from each modal, and gathers to create a TensorDataset
import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

from audio_prec import preprocess_audio


class Input_Features():
    def __init__(self, acoustic_input, visual_input, textual_input, label, unique_id=None):
        self.acoustic_feature = acoustic_input['feature']   # n_frames x feat_dim
        self.acoustic_len = acoustic_input['seq_len']
        self.visual_feature = visual_input['feature']   # n_frames x 3 x dim1 x dim2
        self.textual_input_ids = textual_input['input_ids']     # text_len
        self.textual_attention_mask = textual_input['attention_mask']   # text_len
        self.label = label
        self.unique_id = unique_id  # unique_id to identify examples, used in the test stage


def gather_features(data_type, acoustic_data, visual_data, textual_data, labels):
    """
    gather the features of each example from the three modalities.
    acoustic_data: dict, keys are ids of each example, values are features of each example, each value is dict.
    visual_data, textual_data: dict, similar to acoustic_data
    labels: dict, keys are ids of each example, values are numpy array containing the five labels for each example
    """
    dataset = []
    unique_id = 10000
    id2utter = dict()
    for key in acoustic_data.keys():
        acoustic_input = acoustic_data[key]
        vidual_input = visual_data[key]
        textual_input = textual_data[key]
        label = labels[key]

        dataset.append(Input_Features(acoustic_input, vidual_input, textual_input, label, unique_id))
        unique_id += 1
        id2utter[unique_id] = key
    if data_type == 'test':
        return dataset, id2utter
    return dataset


def prepare_inputs(dataset):
    acoustic_features = torch.FloatTensor([f.acoustic_feature for f in dataset])
    acoustic_lens = torch.FloatTensor([f.acoustic_len for f in dataset])
    visual_features = torch.FloatTensor([f.visual_feature for f in dataset])
    textual_input_ids = torch.LongTensor([f.textual_input_ids for f in dataset])
    textual_attention_mask = torch.LongTensor([f.textual_attention_mask for f in dataset])
    labels = torch.FloatTensor([f.label for f in dataset])
    unique_ids = torch.LongTensor([f.unique_id for f in dataset])

    dataset = TensorDataset(acoustic_features, acoustic_lens, visual_features, textual_input_ids, 
                            textual_attention_mask, labels, unique_ids)
    return dataset


def get_label(data_type):
    """
    read in annotations of data_type, and return a dict
    return: labels: dict, the key is utterance_id and the value is a numpy array containing the label values
    """
    path = "../dataset/raw_data/annotations/"
    filename = os.path.join(path, "annotation_" + data_type + ".pkl")
    with open(filename, "rb") as f:
        annotation = pickle.load(f, encoding='iso-8859-1')
    label_types = annotations.keys()   # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']

    utterance_ids = annotation['extraversion'].keys()
    labels = dict([(utter_id, np.zeros(6)) for utter_id in utterance_ids])
    
    for i, key in enumerate(label_types):
        for utter_id in utterance_ids:
            labels[utter_id][i] = annotation[key][utter_id]
    
    return labels


def save_TensorDataset(dataset, data_type):
    path = "../dataset/preprocessed/"
    filename = os.path.join(path, data_type + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


def prepare_data(data_type):
    """
    read data of data_type from scratch, and preprocess, then change to a TensorDataset
    data_type: str, 'training', 'validation', 'test'
    """
    acoustic_data = preprocess_audio(data_type)  # dict: (file_id, features); features: dict, keys = ('feature', 'seq_len')
    visual_data = preprocess_image(data_type)
    textual_data = preprocess_text(data_type)

    labels = get_label(data_type)   # dict: (utterance_id, label); label: numpy array, size (6)

    if data_type == 'test':
        dataset, id2utter = gather_features(data_type, acoustic_data, visual_data, textual_data, labels)
        path = "../dataset/preprocessed/"
        with open(os.path.join(path, "test_id2utter" + ".pkl")) as f:
            pickle.dump(id2utter, f)
    else:
        dataset = gather_features(data_type, acoustic_data, visual_data, textual_data, labels)
    dataset = prepare_inputs(dataset)   # change to TensorDataset

    save_TensorDataset(dataset, data_type)
    if data_type == 'test':
        return dataset, id2utter
    return dataset