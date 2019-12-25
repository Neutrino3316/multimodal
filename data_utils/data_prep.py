# This file reads in features from each modal, and gathers to create a TensorDataset
import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .audio_prec import preprocess_audio
from .vision_prec import preprocess_image
from .textual_prec import preprocess_text

import pdb

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
        id2utter[unique_id] = key
        unique_id += 1
    if data_type == 'test':
        return dataset, id2utter
    return dataset


def get_real_len(orig_len, args):
    # this function is to compute lens for audio after two conv1d layers
    if orig_len > args.audio_max_frames:
        orig_len = args.audio_max_frames
    kernel_size, stride, padding = args.kernel_size, args.stride, args.padding
    real_len = 1 + (orig_len + 2 * padding - kernel_size) // stride

    real_len = 1 + (real_len + 2 * padding - kernel_size) // stride

    return real_len

def prepare_inputs(args, dataset):
    unique_ids = torch.LongTensor([f.unique_id for f in dataset])

    acoustic_features = torch.FloatTensor([f.acoustic_feature for f in dataset])
    acoustic_features = acoustic_features[:, :, :args.audio_max_frames]
    acoustic_lens = torch.FloatTensor([get_real_len(f.acoustic_len, args) for f in dataset])

    visual_features = torch.stack([f.visual_feature for f in dataset])
    textual_input_ids = torch.stack([f.textual_input_ids for f in dataset])
    textual_input_ids = textual_input_ids.squeeze(1)
    textual_attention_mask = torch.stack([f.textual_attention_mask for f in dataset])
    textual_attention_mask = textual_attention_mask.squeeze(1)

    labels = torch.FloatTensor([f.label for f in dataset])
    extra_token_ids= torch.arange(6).expand(labels.shape[0], 6)
    if not args.interview:
        indices = torch.tensor([0, 1, 2, 3, 5])
        labels = torch.index_select(labels, 1, indices)
        extra_token_ids = torch.index_select(extra_token_ids, 1, indices)

    labels_mask = torch.ones(labels.shape[0], labels.shape[1]).long()
    vision_mask = torch.ones(visual_features.shape[0], visual_features.shape[1]).long()
    acoustic_mask = torch.zeros(acoustic_features.shape[0], get_real_len(args.audio_max_frames, args))
    tmp_acoustic_lens = acoustic_lens.long()
    for i in range(acoustic_features.shape[0]):
        acoustic_mask[i, :tmp_acoustic_lens[i]] = 1
    acoustic_mask = acoustic_mask.long()

    SEP_mask = torch.ones(acoustic_features.shape[0], 1).long()

    fusion_attention_mask = torch.cat([labels_mask, acoustic_mask, SEP_mask, vision_mask, 
                                SEP_mask, textual_attention_mask, SEP_mask], 1)

    dataset = TensorDataset(unique_ids, acoustic_features, acoustic_lens, visual_features, textual_input_ids, 
                            textual_attention_mask, fusion_attention_mask, extra_token_ids, labels)
    return dataset


def get_label(data_type, path):
    """
    read in annotations of data_type, and return a dict
    return: labels: dict, the key is utterance_id and the value is a numpy array containing the label values
    """
    filename = os.path.join(path, "annotation_" + data_type + ".pkl")
    with open(filename, "rb") as f:
        annotations = pickle.load(f, encoding='iso-8859-1')
    label_types = annotations.keys()   # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']

    utterance_ids = annotations['extraversion'].keys()
    labels = dict([(utter_id[:-4], np.zeros(6)) for utter_id in utterance_ids])
    
    for i, key in enumerate(label_types):
        for utter_id in utterance_ids:
            labels[utter_id[:-4]][i] = annotations[key][utter_id]
    
    return labels


def save_Dataset(dataset, data_type, path):
    # path = "../dataset/preprocessed/"
    filename = os.path.join(path, data_type + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


def save_modal_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_modal_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def prepare_data(data_type, data_path):
    """
    read data of data_type from scratch, and preprocess
    data_type: str, 'training', 'validation', 'test'
    """
    filename = os.path.join(os.path.join(data_path, "tmp/"), f"audio_{data_type}.pkl")
    if os.path.exists(filename):
        acoustic_data = load_modal_data(filename)
    else:    
        acoustic_data = preprocess_audio(data_type, os.path.join(data_path, 'raw_data/audio/'))  # dict: (file_id, features); features: dict, keys = ('feature', 'seq_len')    
        save_modal_data(acoustic_data, filename)

    filename = os.path.join(os.path.join(data_path, "tmp/"), f"vision_{data_type}.pkl")
    if os.path.exists(filename):
        visual_data = load_modal_data(filename)
    else:
        visual_data = preprocess_image(data_type, os.path.join(data_path, 'raw_data/vision/'))
        save_modal_data(visual_data, filename)
    
    filename = os.path.join(os.path.join(data_path, "tmp/"), f"text_{data_type}.pkl")
    if os.path.exists(filename):
        textual_data = load_modal_data(filename)
    else:
        textual_data = preprocess_text(data_type, os.path.join(data_path, 'raw_data/text/'))
        save_modal_data(textual_data, filename)

    labels = get_label(data_type, os.path.join(data_path, "annotations/"))   # dict: (utterance_id, label); label: numpy array, size (6)

    path = os.path.join(data_path, "preprocessed/")
    if data_type == 'test':
        dataset, id2utter = gather_features(data_type, acoustic_data, visual_data, textual_data, labels)
        with open(os.path.join(path, "test_id2utter" + ".pkl"), "wb") as f:
            pickle.dump(id2utter, f)
    else:
        dataset = gather_features(data_type, acoustic_data, visual_data, textual_data, labels)
    
    save_Dataset(dataset, data_type, path)

    if data_type == 'test':
        return dataset, id2utter
    return dataset