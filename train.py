import argparse
import pickle
import os
import random
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

from data_utils.data_prep import prepare_data
from models.model import TriModalModel

import logging
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # file and directory settings
    parser.add_argument('--exp_name', type=str, required=True, help="name of the experiment, required to save checkpoint"
                                                                    "and training details")

    # model settings
    ## overal settings
    parser.add_argument('--interview', action="store_true", help="whether to use")
    parser.add_argument('--out_dim', type=int, default=768, help="dimension of features before fusion")
    ## AudioModel settings
    parser.add_argument('--audio_n_gru', type=int, default=2, help="number of gru layers")
    ## VisionModel settings
    parser.add_argument('--vision_n_lstm', type=int, default=1, help="number of lstm layers")
    parser.add_argument('--vgg_param_dir', type=str, default="./dataset/pretrained_models/vgg_face_dag.pth")
    ## TextModel settings
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument('--textual_model_type', type=str, required=True, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--config_name', default="", type=str,
                        help="Pretrained config name or path if not the same as model_name"))
    parser.add_argument('--cache_dir', default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    ## FusionModel settings
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help="dropout rate for transformer")
    parser.add_argument('--n_fusion_layers', default=6, type=int, help="number of encoder layers for fusion module")
    parser.add_argument('--fusion_hid_dim', default=768, type=int, help="hidden size for encoder layers in fusion module")
    parser.add_argument('--n_attn_heads', default=8, type=int, help="number of heads for attention")
    parser.add_argument('--fusion_ffw_dim', default=2048, type=int, help="dim for feed forward layers in encoder")

    # experiment settings
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=5e-5, help="initial learning rate")
    parser.add_argument('--wd', type=float, default=0.0, help="weight decay")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--warmup_steps', default=2000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=1212)

    # additional settings
    parser.add_argument()
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

        self.n_valid, self.n_test = len(validset), len(testset)
        self.train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
        
        model.to(device)
        self.model = model

        self.args.total_steps = len(train_loader) // args.accum_steps * args.n_epochs
        logger.info("Total training steps: %d" % total_steps)

        # Prepare opitimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

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
                'labels': batch[7]}
            outputs = model(**inputs)
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
                self.scheduler.step()
                self.model.zero_grad()
                global_step += 1

                if (global_step + 1) % self.args.log_interval == 0:
                    lr = self.scheduler.get_lr()[0]
                    cur_loss = (tr_loss - logging_loss) / self.args.log_interval
                    self.logger.info(f"Epoch: {epoch}; step {step}; lr {lr:.5f}; loss {cur_loss:.5f}")
                    logging_loss = tr_loss
        
        avg_loss = tr_loss / (step+1) * self.args.accum_steps
        self.logger.info(f"-----Average loss for Epoch {epoch}: {avg_loss:.5f}-----")

        return global_step

    def valid_one_epoch(self, epoch):
        self.model.eval()
        self.logger.info("***** Running Validation *****")
        eval_loss = 0.0
        error = np.zeros(self.model.num_labels)
        for batch in tqdm(self.valid_loader, desc="Validation"):
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {'audio_feature': batch[1], 'audio_len': batch[2], 'vision_feature': batch[3], 
                'text_input_ids': batch[4], 'text_attn_mask': batch[5], 'fusion_attn_mask': batch[6], 
                'labels': batch[7]}

            tmp_valid_loss, logits = model(**inputs)
            valid_loss += tmp_valid_loss.mean().item()
            tmp_err = torch.sum(torch.abs(inputs['labels'], logits), dim=0)
            error = error + err.detach().cpu().numpy()

        error = error / self.n_valid
        accuracy = 1 - error
        avg_acc = np.mean(accuracy)

        log_str = f"Epoch {epoch}; accuracy {avg_acc}: "
        for i, name in enumerate(self.labels_names):
            log_str += f"{name} {accuracy[i]}/"
        self.logger.info(log_str)

        return avg_acc

    def test(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(checkpoint_dir))
        self.logger.info("***** Start Testing *****")
        self.logger.info(f"Loaded model checkpoint from {checkpoint_dir}")
        self.model.eval()

        predictions = dict()
        test_loss = 0.
        error = np.zeros(self.n_labels)
        for batch in tqdm(self.test_loader, desc="Testing"):
            batch = tuple(t.to(args.device) for t in batch)

            unique_id = batch[0]
            inputs = {'audio_feature': batch[1], 'audio_len': batch[2], 'vision_feature': batch[3], 
                'text_input_ids': batch[4], 'text_attn_mask': batch[5], 'fusion_attn_mask': batch[6], 
                'labels': batch[7]}
            tmp_test_loss, logits = model (**inputs)
            test_loss += tmp_test_loss.mean().item()
            tmp_err = torch.sum(torch.abs(inputs['labels'], logits), dim=0)
            error = error + err.detach().cpu().numpy()

            preds = logits.detach().cpu().numpy()
            for i, id in enumerate(unique_id.detach().cpu().numpy().tolist()):
                cur_pred = preds[i].tolist()
                predictions[self.id2utter[id]] = dict()
                for i, name in enumerate(self.labels_names):
                    predictions[self.id2utter[id]][name] = cur_pred[i]
        
        error = error / self.n_test
        accuracy = 1 - error
        avg_acc = np.mean(accuracy)

        log_str = f"Epoch {epoch}; accuracy {avg_acc}: "
        for i, name in enumerate(self.labels_names):
            log_str += f"{name} {accuracy[i]}/"
        self.logger.info(log_str)

        out_dir = os.path.join("./snapshots/", self.args.exp_name)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)


    def save_checkpoint(self, epoch):
        out_dir = os.path.join("./snapshots/", self.args.exp_name)
        output_model_file = os.path.join(out_dir, f"e_{epoch}.pt")
        torch.save(self.model.state_dict(), output_model_file)
        self.logger.info(f"Saving checkpoint to {output_model_file}.")


def remove_useless_checkpoint(out_dir, best_pt):
    all_files = os.listdir(out_dir)
    pt_files = list(filter(lambda file: file.split(".")[-1] == "pt", all_files))
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

    data_dir = os.path.join(args.data_path, "preprocessed/")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        trainset, validset, testset, id2utter = load_data()
    else:
        trainset, validset = prepare_data('training'), prepare_data('validation')
        testset, id2utter = prepare_data('test')

    model = TriModalModel(args)
    trainer = TriModalTrainer(args, logger, model, trainset, validset, testset, 
                            labels_names, id2utter)

    accuracy = trainer.train()

    best_epoch = np.argmax(accuracy)
    input_model_file = os.path.join(out_dir, f"e_{best_epoch}.pt")

    audio_trainer.test(input_model_file)
    remove_useless_checkpoint(out_dir, input_model_file)