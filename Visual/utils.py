import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_iter(files, labels, batch_size):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    """
    batch_num = math.ceil(len(files) / batch_size)
    index_array = list(range(len(files)))

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [files[idx] for idx in indices]
        train_labels = {}
        for file in examples:
            train_labels[file + '.mp4'] = labels[file + '.mp4']
        train_files = examples

        yield train_files, train_labels
