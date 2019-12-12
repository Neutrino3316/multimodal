import argparse
import pickle
import os

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


if __name__ == '__main__':
    args = get_args()                                                                    