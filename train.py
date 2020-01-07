import argparse
import pickle
import os
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

from data_utils import prepare_data, prepare_inputs
from models import TriModalModel, AvgModalModel

import pdb

import logging
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # file and directory settings
    parser.add_argument('--exp_name', type=str, required=True, help="name of the experiment, required to save checkpoint"
                                                                    "and training details")
    parser.add_argument('--data_path', type=str, default="./dataset/", help="directory to data")

    # model settings
    ## overal settings
    parser.add_argument('--model_type', type=str, default="TriModalModel", help="options: ['TriModalModel', 'AvgModalModel']")
    parser.add_argument('--interview', action="store_true", help="whether to use")
    parser.add_argument('--out_dim', type=int, default=768, help="dimension of features before fusion")
    ## AudioModel settings
    parser.add_argument('--audio_max_frames', type=int, default=600, help="max frames for audio")
    parser.add_argument('--audio_n_gru', type=int, default=2, help="number of gru layers")
    parser.add_argument('--kernel_size', type=int, default=5, help="conv kernel size for audiomodel")
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--stride', type=int, default=2)
    ## VisionModel settings
    parser.add_argument('--pretrained_resnet', action='store_true')
    parser.add_argument('--vision_n_gru', type=int, default=1, help="number of lstm layers")
    parser.add_argument('--vgg_param_dir', type=str, default="./pretrained_weights/vgg_face_dag.pth")
    ## TextModel settings
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument('--cache_dir', default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    ## FusionModel settings
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help="dropout rate for transformer")
    parser.add_argument('--n_fusion_layers', default=6, type=int, help="number of encoder layers for fusion module")
    parser.add_argument('--fusion_hid_dim', default=768, type=int, help="hidden size for encoder layers in fusion module")
    parser.add_argument('--n_attn_heads', default=4, type=int, help="number of heads for attention")
    parser.add_argument('--fusion_ffw_dim', default=2048, type=int, help="dim for feed forward layers in encoder")

    # experiment settings
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=5e-5, help="initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--warmup_steps', default=2000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=1212)

    # additional settings
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class TriModalTrainer():
    def __init__(self, args, logger, model, trainset, validset, testset, 
            labels_names, id2utter):
        self.args = args
        self.labels_names = labels_names
        self.id2utter = id2utter

        self.logger=logger

        self.n_valid, self.n_test = len(validset), len(testset)
        self.train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
        
        self.model = model
        self.model.to(args.device)

        self.args.total_steps = len(self.train_loader) // args.accum_steps * args.n_epochs
        logger.info("Total training steps: %d" % self.args.total_steps)

        # Prepare opitimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=self.args.total_steps)

        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def train(self):
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", args.n_epochs)
        logger.info("  Gradient Accumulation steps = %d", args.accum_steps)
        logger.info("  Total train batch size = %d", args.batch_size)
        logger.info("  Total optimization steps = %d", self.args.total_steps)

        accuracy = np.zeros(self.args.n_epochs)
        global_step = 0
        set_seed(self.args)     # Added here for reproductibility (even between python 2 and 3)
        for epoch in trange(int(self.args.n_epochs), desc="Epoch"):
            global_step += self.train_one_epoch(epoch, global_step)
            acc = self.valid_one_epoch(epoch)
            accuracy[epoch] = acc
            self.save_checkpoint(epoch)
        return accuracy
    
    def train_one_epoch(self, epoch, global_step):
        self.model.train()
        tr_loss, logging_loss = 0., 0.
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {'audio_feature': batch[1], 'audio_len': batch[2], 'vision_feature': batch[3], 
                'text_input_ids': batch[4], 'text_attn_mask': batch[5], 'fusion_attn_mask': batch[6], 
                'extra_token_ids': batch[7], 'labels': batch[8]}
            outputs = self.model(**inputs)
            loss = outputs[0]

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.accum_steps > 1:
                loss = loss / self.args.accum_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % self.args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()
                # self.scheduler.step()
                self.model.zero_grad()
                global_step += 1

                if (global_step + 1) % self.args.log_interval == 0:
                    # lr = self.scheduler.get_lr()[0]
                    cur_loss = (tr_loss - logging_loss) / self.args.log_interval
                    self.logger.info(f"Epoch: {epoch}; step {step}; lr {self.args.lr:.6f}; loss {cur_loss:.6f}")
                    logging_loss = tr_loss
        
        avg_loss = tr_loss / (step+1) * self.args.accum_steps
        self.logger.info(f"-----Average loss for Epoch {epoch}: {avg_loss:.6f}-----")

        return global_step

    def valid_one_epoch(self, epoch):
        self.model.eval()
        self.logger.info("***** Running Validation *****")
        valid_loss = 0.0
        error = np.zeros(len(self.labels_names))
        for batch in tqdm(self.valid_loader, desc="Validation"):
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {'audio_feature': batch[1], 'audio_len': batch[2], 'vision_feature': batch[3], 
                'text_input_ids': batch[4], 'text_attn_mask': batch[5], 'fusion_attn_mask': batch[6], 
                'extra_token_ids': batch[7], 'labels': batch[8]}

            tmp_valid_loss, logits, _ = self.model(**inputs)
            valid_loss += tmp_valid_loss.mean().item()
            tmp_err = torch.sum(torch.abs(inputs['labels'] - logits), dim=0)
            error = error + tmp_err.detach().cpu().numpy()

        error = error / self.n_valid
        accuracy = 1 - error
        avg_acc = np.mean(accuracy)
        valid_loss = valid_loss / len(self.valid_loader)

        log_str = f"Validation: epoch {epoch}; loss {valid_loss:.6f}; accuracy {avg_acc:.6f}: "
        for i, name in enumerate(self.labels_names):
            log_str += f"{name} {accuracy[i]:.6f}/"
        self.logger.info(log_str)

        return avg_acc

    def test(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(checkpoint_dir))
        self.logger.info("***** Start Testing *****")
        self.logger.info(f"Loaded model checkpoint from {checkpoint_dir}")
        self.model.eval()

        predictions = dict()
        attention_scores = []
        test_loss = 0.
        error = np.zeros(len(self.labels_names))
        for batch in tqdm(self.test_loader, desc="Testing"):
            batch = tuple(t.to(args.device) for t in batch)

            unique_id = batch[0]
            inputs = {'audio_feature': batch[1], 'audio_len': batch[2], 'vision_feature': batch[3], 
                'text_input_ids': batch[4], 'text_attn_mask': batch[5], 'fusion_attn_mask': batch[6], 
                'extra_token_ids': batch[7], 'labels': batch[8]}
            tmp_test_loss, logits, attns = self.model(**inputs)
            attention_scores.append([attn.detach().cpu().numpy() for attn in attns])
            test_loss += tmp_test_loss.mean().item()
            tmp_err = torch.sum(torch.abs(inputs['labels'] - logits), dim=0)
            error = error + tmp_err.detach().cpu().numpy()

            preds = logits.detach().cpu().numpy()
            for i, id in enumerate(unique_id.detach().cpu().numpy().tolist()):
                cur_pred = preds[i].tolist()
                predictions[self.id2utter[id]] = dict()
                for i, name in enumerate(self.labels_names):
                    predictions[self.id2utter[id]][name] = cur_pred[i]
        
        error = error / self.n_test
        accuracy = 1 - error
        avg_acc = np.mean(accuracy)
        test_loss = test_loss / len(self.test_loader)

        log_str = f"Test loss: {test_loss:.6f}; accuracy {avg_acc:.6f}: "
        for i, name in enumerate(self.labels_names):
            log_str += f"{name} {accuracy[i]:.6f}/"
        self.logger.info(log_str)

        out_dir = os.path.join("./snapshots/", self.args.exp_name)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        
        with open(os.path.join(out_dir, "attn_scores.pkl"), "wb") as f:
            pickle.dump(attention_scores, f)


    def save_checkpoint(self, epoch):
        out_dir = os.path.join("./snapshots/", self.args.exp_name)
        output_model_file = os.path.join(out_dir, f"e_{epoch}.pt")
        torch.save(self.model.state_dict(), output_model_file)
        self.logger.info(f"Saving checkpoint to {output_model_file}.")


def remove_useless_checkpoint(out_dir, best_pt):
    all_files = os.listdir(out_dir)
    pt_files = list(filter(lambda file: file.split(".")[-1] == "pt", all_files))

    pt_files = [os.path.join(out_dir, ptfile) for ptfile in pt_files]
    pt_files.remove(best_pt)
    for f in pt_files:
        os.remove(f)



if __name__ == '__main__':
    args = get_args()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    out_dir = os.path.join("./snapshots/", args.exp_name)
    if os.path.exists(out_dir):
        raise ValueError("Output directory ({}) already exists.".format(out_dir))
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

    data_dir = os.path.join("./dataset/", "preprocessed/")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        trainset, validset, testset, id2utter = load_data()
    else:
        validset, trainset = prepare_data('validation', args.data_path), prepare_data('training', args.data_path)
        testset, id2utter = prepare_data('test', args.data_path)

    trainset, validset, testset = prepare_inputs(args, trainset), \
        prepare_inputs(args, validset), prepare_inputs(args, testset)   # change to TensorDataset

    model_dict = {'TriModalModel': TriModalModel, 'AvgModalModel': AvgModalModel}
    model = model_dict[args.model_type](args)
    # model = TriModalModel(args)
    trainer = TriModalTrainer(args, logger, model, trainset, validset, testset, 
                            labels_names, id2utter)

    accuracy = trainer.train()

    best_epoch = np.argmax(accuracy)
    input_model_file = os.path.join(out_dir, f"e_{best_epoch}.pt")

    trainer.test(input_model_file)
    remove_useless_checkpoint(out_dir, input_model_file)