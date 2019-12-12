import argparse
import pickle
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.data_prep import prepare_data

import logging
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # file and directory settings
    parser.add_argument('--exp_name', type=str, required=True, help="name of the experiment, required to save checkpoint"
                                                                    "and training details")

    # model settings
    parser.add_argument('--interview', action="store_true", help="whether to use")

    # experiment settings
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0, help="weight decay")
    parser.add_argument('--seed', type=int, default=1212)

    # additional settings
    parser.add_argument('--loss_exp', type=float, default=0.9, help="exponential loss averaging")
    parser.add_argument('--log_interval', type=int, default=10, help="steps to log training information")

    args = parser.parse_args()
    print(args)
    return args


def load_data():
    path = "./dataset/preprocessed/"
    with open(os.path.join(path, "training.pkl"), "rb") as f:
        trainset = pickle.load(f)
    with open(os.path.join(path, "validation.pkl"), "rb") as f:
        validset = pickle.load(f)
    with open(os.path.join(path, "test.pkl"), "rb") as f:
        testset = pickle.load(f)
    with open(os.path.join(path, "test_id2utter.pkl"), "rb") as f:
        id2utter = pickle.load(f)
    return trainset, validset, testset, id2utter


def build_model(args):



if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    out_dir = os.path.join("../snapshots/", args.exp_name)
    if os.path.exists(out_dir):
        raise ValueError("Output directory () already exists.")
    else:
        os.makedirs(out_dir)

    log_file = os.path.join(out_dir, "log.log")
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(args)

    n_labels = 6 if args.interview else 5
    labels_names = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    if n_labels == 5:
        labels_names.remove('interview')

    data_dir = os.path.join(args.data_path, "preprocessed/")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        trainset, validset, testset, id2utter = load_data()
    else:
        trainset, validset = prepare_data('training'), prepare_data('validation')
        testset, id2utter = prepare_data('test')

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args)