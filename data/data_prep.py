# This file reads in features from each modal, and gathers to create a TensorDataset

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


def gather_features(acoustic_data, visual_data, textual_data, labels):
    """
    gather the features of each example from the three modalities.
    acoustic_data: dict, keys are ids of each example, values are features of each example, each value is dict.
    visual_data, textual_data: dict, similar to acoustic_data
    labels: dict, keys are ids of each example, values are numpy array containing the five labels for each example
    """
    dataset = []
    unique_id = 10000
    for key in acoustic_data.keys():
        acoustic_input = acoustic_data[key]
        vidual_input = visual_data[key]
        textual_input = textual_data[key]
        label = labels[key]

        dataset.append(Input_Features(acoustic_input, vidual_input, textual_input, label, unique_id))
        unique_id += 1

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
    #TODO


def save_TensorDataset(dataset, data_type):
    #TODO


def prepare_data(data_type):
    """
    read data of data_type from scratch, and preprocess, then change to a TensorDataset
    data_type: str, 'training', 'validation', 'test'
    """
    acoustic_data = preprocess_audio(data_type)  # dict: (file_id, features); features: dict, keys = ('feature', 'seq_len')
    visual_data = preprocess_image(data_type)
    textual_data = preprocess_text(data_type)

    labels = get_label(data_type)

    dataset = gather_features(acoustic_data, visual_data, textual_data, labels)
    dataset = prepare_inputs(dataset)   # change to TensorDataset

    save_TensorDataset(dataset, data_type)
    return dataset