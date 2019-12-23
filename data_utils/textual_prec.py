from __future__ import absolute_import, division, print_function

import os
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange



from transformers import BertTokenizer
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features


model_path = "./pretrained_weights/" # Bert model path

#Name不用单独放在一个文件夹了
def load_and_cache_examples(task, tokenizer, datatype, path, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    if datatype == 'training':
        examples, Video_id = processor.get_train_examples(path) # path file : postaddress must be .tsv
    elif datatype == 'validation':
        examples, Video_id = processor.get_dev_examples(path)
    elif datatype == 'test':
        examples, Video_id = processor.get_test_examples(path) 

    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list = label_list,
                                            max_length = 128,
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
        data[ Video_id[index] ] = {'input_ids': input_ids, 'attention_mask': attention_mask}

    return data

# Call. fuc.
def preprocess_text(datatype, data_path):
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=1, cache_dir=None)
    data = load_and_cache_examples('pedt', tokenizer, datatype, data_path, evaluate=False)
    return data


if __name__ == '__main__':
    import pdb
    path = "../dataset/raw_data/text/"
    dataset = preprocess_text('test', path)
    pdb.set_trace()