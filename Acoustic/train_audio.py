import argparse
import os
import pickle
from tqdm import tqdm, trange
import logging
logger = logging.getLogger(__name__)
import collections
import json
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal

from dataset_audio import preprocess
from model_audio import audio_model

def get_args():
    parser = argparse.ArgumentParser()
    # file and directory settings
    parser.add_argument('--data_path', type=str, default='../dataset/',
                        help="directory for annotations and preprocessed data")
    parser.add_argument('--exp_name', type=str, required=True, help="name of the experiment, required to save checkpoint"
                                                                    "and training details")

    # model settings
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of gru layers")
    parser.add_argument('--out_dim', type=int, default=768, help="dim of gru outputs")
    parser.add_argument('--interview', action='store_true', help="whether to use the interview subtask")

    # experiment settings
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0, help="weight decay")
    parser.add_argument('--seed', type=int, default=1118)

    parser.add_argument('--uniform_labels', action='store_true', help="whether to change the labels so as to yield uniform distributions")

    # additional settings
    parser.add_argument('--loss_exp', type=float, default=0.9, help="exponential loss averaging")
    parser.add_argument('--log_interval', type=int, default=10, help="steps to log training information")

    args = parser.parse_args()
    print(args)
    return args

#
# RawResult = collections.namedtuple("RawResult", ['extraversion', 'neuroticism', 'agreeableness', 'interview',
#                                                  'conscientiousness', 'openness'])


def load_data(data_path):
    with open(os.path.join(data_path, "preprocessed/preproc_audio_train.pkl"), "rb") as f:
        trainset = pickle.load(f)
    with open(os.path.join(data_path, "preprocessed/preproc_audio_valid.pkl"), "rb") as f:
        validset = pickle.load(f)
    with open(os.path.join(data_path, "preprocessed/preproc_audio_test.pkl"), "rb") as f:
        testset = pickle.load(f)
    with open(os.path.join(data_path, "preprocessed/idx_dict.pkl"), "rb") as f:
        idx2name = pickle.load(f)

    return trainset, validset, testset, idx2name


def transform_labels(true_labels):
    pseudo_labels = (true_labels - labels_mean) / labels_std
    dist = Normal(0, 1)
    pseudo_labels = dist.cdf(pseudo_labels)
    return pseudo_labels


def inverse_transform_labels(pseudo_labels):
    dist = Normal(0, 1)
    true_labels = dist.icdf(pseudo_labels)
    true_labels = true_labels * labels_std + labels_mean
    return true_labels


class labels_transformer():
    def __init__(self):
        self.labels_mean = torch.FloatTensor(
            [0.4761464174454829, 0.5202864583333333, 0.5481813186813186, 0.5227313915857604, 0.5037803738317757,
             0.5662814814814815])
        self.labels_std = torch.FloatTensor(
            [0.15228452985134602, 0.15353347248058757, 0.13637365282783034, 0.15520650375390665, 0.15013557786759546,
             0.14697755975897248])
        self.dist = Normal(0, 1)

    def transform_labels(self, true_labels):
        pseudo_labels = (true_labels - self.labels_mean) / self.labels_std
        pseudo_labels = self.dist.cdf(pseudo_labels)
        return pseudo_labels

    def inverse_transform_labels(self, pseudo_labels):
        true_labels = self.dist.icdf(pseudo_labels)
        true_labels = true_labels * self.labels_std + self.labels_mean
        return true_labels


def prepare_data(dataset, n_labels=6, uniform_labels=False, for_test=False, transformer=None):
    features = torch.FloatTensor([f.feature for f in dataset])
    labels = torch.FloatTensor([f.label for f in dataset])
    if uniform_labels:
        pseudo_labels = transformer.transform_labels(labels)
    seq_lens = torch.FloatTensor([f.seq_len for f in dataset])
    if n_labels == 5:
        indices = torch.tensor([0, 1, 2, 3, 5])
        labels = torch.index_select(labels, 1, indices)
        if uniform_labels:
            pseudo_labels = torch.index_select(pseudo_labels, 1, indices)
    if for_test:
        unique_dix = torch.LongTensor([f.unique_idx for f in dataset])
        if uniform_labels:
            dataset = TensorDataset(features, seq_lens, pseudo_labels, labels, unique_dix)
        else:
            dataset = TensorDataset(features, seq_lens, labels, unique_dix)
    else:
        if uniform_labels:
            dataset = TensorDataset(features, seq_lens, pseudo_labels, labels)
        else:
            dataset = TensorDataset(features, seq_lens, labels)
    return dataset


class trainer():
    def __init__(self, args, train_loader, valid_loader, test_loader, model, labels_names, idx2name,
                 logger, n_gpu, device, n_train, n_valid, n_test, transformer=None):
        self.args = args
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.model = model
        self.labels_names = labels_names
        self.n_labels = len(labels_names)
        self.idx2name = idx2name
        self.logger = logger
        self.n_gpu = n_gpu
        self.n_train, self.n_valid, self.n_test = n_train, n_valid, n_test
        if args.uniform_labels:
            self.transformer = transformer

        self.model.to(device)
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def train(self):
        accuracy = np.zeros(self.args.n_epochs)
        for epoch in trange(int(self.args.n_epochs), desc="Epoch"):
            self.train_one_epoch(epoch)
            acc = self.valid_one_epoch(epoch)
            accuracy[epoch] = acc
            self.save_checkpoint(epoch)
        return accuracy

    def train_one_epoch(self, epoch):
        self.model.train()
        avg_loss = 0.
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            if self.n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self

            if self.args.uniform_labels:
                features, seq_lens, labels, true_labels = batch     # labels are pseudo_labels
            else:
                features, seq_lens, labels = batch  # labels are true_labels
            loss = self.model(features, seq_lens, labels)

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if self.args.accum_steps > 1:
                loss = loss / self.args.accum_steps
            avg_loss = self.args.loss_exp * avg_loss + (1 - self.args.loss_exp) * loss.item()

            loss.backward()

            if (step + 1) % self.args.accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if step % self.args.log_interval == 0:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                log_str = f" Epoch {epoch}; step {step}; lr {lr:.5f}; loss {avg_loss:.5f}"
                self.logger.info(log_str)

    def valid_one_epoch(self, epoch):
        self.model.eval()
        self.logger.info("Start Validation ------------")
        error = np.zeros(self.n_labels)
        true_error = np.zeros(self.n_labels)

        for batch in tqdm(self.valid_loader, desc="Validation"):
            if self.n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            if self.args.uniform_labels:
                features, seq_lens, labels, true_labels = batch     # labels are pseudo_labels
            else:
                features, seq_lens, labels = batch  # labels are true_labels

            preds = self.model(features, seq_lens)  # pseudo preds
            err = torch.sum(torch.abs(labels - preds), dim=0)
            error = error + err.detach().cpu().numpy()
            if self.args.uniform_labels:
                tmp_preds = preds.detach().cpu()
                true_labels = true_labels.detach().cpu()
                true_preds = self.transformer.inverse_transform_labels(tmp_preds)
                true_err = torch.sum(torch.abs(true_labels - true_preds), dim=0)
                true_error = true_error + true_err.numpy()
        error = error / self.n_valid
        accuracy = 1 - error
        avg_acc = np.mean(accuracy)
        log_str = f"Epoch {epoch}; accuracy {avg_acc}: "
        for i, name in enumerate(self.labels_names):
            log_str += f"{name} {accuracy[i]}/"
        self.logger.info(log_str)

        if self.args.uniform_labels:
            true_error = true_error / self.n_valid
            true_accuracy = 1 - true_error
            true_avg_acc = np.mean(true_accuracy)
            log_str = f"Epoch {epoch}; true acc {true_avg_acc}: "
            for i, name in enumerate(self.labels_names):
                log_str += f"{name} {true_accuracy[i]}/"
            self.logger.info(log_str)
            return true_avg_acc
        return avg_acc

    def test(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(checkpoint_dir))
        self.logger.info(f"Loaded model checkpoint from {checkpoint_dir}")
        self.model.eval()
        self.logger.info("Start Testing --------------")
        error = np.zeros(self.n_labels)
        true_error = np.zeros(self.n_labels)

        predictions, true_predictions = dict(), dict()
        for batch in tqdm(self.test_loader, desc="Testing"):
            if self.n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            if self.args.uniform_labels:
                features, seq_lens, labels, true_labels, unique_idx = batch     # labels are pseudo_labels
            else:
                features, seq_lens, labels, unique_idx = batch  # labels are true_labels
            preds = self.model(features, seq_lens)  # batch x n_labels
            err = torch.sum(torch.abs(labels - preds), dim=0)
            error = error + err.detach().cpu().numpy()

            if self.args.uniform_labels:
                tmp_preds = preds.detach().cpu()
                true_labels = true_labels.detach().cpu()
                true_preds = self.transformer.inverse_transform_labels(tmp_preds)
                true_err = torch.sum(torch.abs(true_labels - true_preds), dim=0)
                true_error = true_error + true_err.numpy()

            preds = preds.detach().cpu().numpy()
            for i, id in enumerate(unique_idx.detach().cpu().numpy().tolist()):
                cur_pred = preds[i].tolist()
                predictions[self.idx2name[id]] = dict()
                for i, name in enumerate(self.labels_names):
                    predictions[self.idx2name[id]][name] = cur_pred[i]

            if self.args.uniform_labels:
                true_preds = true_preds.numpy()
                for i, id in enumerate(unique_idx.detach().cpu().numpy().tolist()):
                    cur_pred = true_preds[i].tolist()
                    true_predictions[self.idx2name[id]] = dict()
                    for i, name in enumerate(self.labels_names):
                        true_predictions[self.idx2name[id]][name] = cur_pred[i]

        error = error / self.n_test
        accuracy = 1 - error
        avg_acc = np.mean(accuracy)
        log_str = f"Test Accuracy {avg_acc}: "
        for i, name in enumerate(self.labels_names):
            log_str += f"{name} {accuracy[i]}/"
        self.logger.info(log_str)
        out_dir = os.path.join("../snapshots/", self.args.exp_name)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)

        if self.args.uniform_labels:
            true_error = true_error / self.n_test
            true_accuracy = 1 - true_error
            true_avg_acc = np.mean(true_accuracy)
            log_str = f"Test true acc {true_avg_acc}: "
            for i, name in enumerate(self.labels_names):
                log_str += f"{name} {true_accuracy[i]}/"
            self.logger.info(log_str)
            out_dir = os.path.join("../snapshots/", self.args.exp_name)
            with open(os.path.join(out_dir, "true_predictions.json"), "w") as f:
                json.dump(true_predictions, f)

    def save_checkpoint(self, epoch):
        out_dir = os.path.join("../snapshots/", self.args.exp_name)
        output_model_file = os.path.join(out_dir, f"e_{epoch}.pt")
        torch.save(self.model.state_dict(), output_model_file)
        logger.info(f"Saving checkpoint to {output_model_file}.")


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

    transformer = None
    if args.uniform_labels:
        transformer = labels_transformer()

    data_dir = os.path.join(args.data_path, "preprocessed/")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        trainset, validset, testset, idx2name = load_data(args.data_path)
    else:
        trainset, validset, testset, idx2name = preprocess()
    trainset, validset, testset = prepare_data(dataset=trainset, n_labels=n_labels, uniform_labels=args.uniform_labels, for_test=False, transformer=transformer), \
                                  prepare_data(dataset=validset, n_labels=n_labels, uniform_labels=args.uniform_labels, for_test=False, transformer=transformer), \
                                  prepare_data(dataset=testset, n_labels=n_labels, uniform_labels=args.uniform_labels, for_test=True, transformer=transformer)

    if args.uniform_labels and n_labels == 5:
        indices = torch.tensor([0, 1, 2, 3, 5])
        transformer.labels_mean = torch.index_select(transformer.labels_mean, 0, indices)
        transformer.labels_std = torch.index_select(transformer.labels_std, 0, indices)

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    model = audio_model(args)

    audio_trainer = trainer(args, train_loader, valid_loader, test_loader, model,
                            labels_names, idx2name, logger, n_gpu, device,
                            len(trainset), len(validset), len(testset), transformer)

    accuracy = audio_trainer.train()

    best_epoch = np.argmax(accuracy)
    input_model_file = os.path.join(out_dir, f"e_{best_epoch}.pt")

    audio_trainer.test(input_model_file)
