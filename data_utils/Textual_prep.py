from __future__ import absolute_import, division, print_function

import os
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

path = "../dataset/raw_data/text/" # temporary path, text.tsv / dev.tsv
namefile = "../dataset/raw_data/test/ViedoName.txt" # an extra name file
model_path = "./model/" # Bert model path

def load_and_cache_examples(task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    with open(namefile, 'r') as file:
        Video_id = file.readlines()

    examples = processor.get_train_examples(path) # path file : postaddress must be .tsv
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list = label_list,
                                            max_length = 256,
                                            output_mode = output_mode,
                                            pad_on_left = 0,   # pad on the left for xlnet
                                            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id = 0,
    )

    # Convert to Tensors
    data = dict()
    for index, f in enumerate(features):
        input_ids = torch.tensor([f.input_ids], dtype = torch.long)
        attention_mask = torch.tensor([f.attention_mask], dtype=torch.long)
        data[ Video_id[index] ] = {'input_ids':input_ids, 'attention_mask':attention_mask}

    return data


# Call. fuc.
def preprocess_text(datatype):
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case = 1, cache_dir = None)
    data = load_and_cache_examples('pedt', tokenizer, evaluate=False)
    return data
