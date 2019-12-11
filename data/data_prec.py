# This file reads in features from each modal, and gathers to create a TensorDataset

class Input_Features():
    def __init__(self, acoustic_input, visual_input, textual_input, label, unique_id=None):
        self.acoustic_feature = acoustic_input['feature']   # n_frames x feat_dim
        self.acoustic_len = acoustic_input['seq_len']
        self.visual_feature = visual_input['feature']   # n_frames x 3 x dim1 x dim2
        self.textual_input_ids = textual_input['input_ids']     # text_len
        self.textual_attention_mask = textual_input['attention_mask']   # text_len
        self.label = label
        if unique_id is not None:
            self.unique_id = unique_id  # unique_id to identify examples, used in the test stage


def gather_features(acoustic_data, visual_data, textual_data):
    """
    gather the features of each example from the three modalities.
    acoustic_data: dict, keys are ids of each example, values are features of each example, each value is dict.
    visual_data, textual_data: dict, similar to acoustic_data
    """
    dataset = []
    for key in acoustic_data.keys():
        acoustic_input = acoustic_data[key]
        vidual_input = visual_data[key]
        textual_input = textual_data[key]
        label = acoustic_data[key]['label']
        unique_id = acoustic_data[key]['unique_id']

        dataset.append(Input_features(acoustic_input, vidual_input, textual_input, label, unique_id))

    return dataset
