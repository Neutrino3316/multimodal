import torch
import torch.nn as nn

from torchvision.models import resnet18

from .Vgg_face import build_vgg
from .transformer import Encoder, GELU

import math
import sys
sys.path.append("..")
from transformers import BertModel

import pdb


class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        input_dim, out_dim, n_gru_layers = 68, args.out_dim, args.audio_n_gru
        kernel_size, stride, padding = args.kernel_size, args.stride, args.padding
        # n_cls = 6 if args.interview else 5

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=out_dim//2, kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm1d(out_dim // 2), 
                                    GELU(),
                                    nn.Dropout(args.dropout))

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=out_dim//2, out_channels=out_dim*2, kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm1d(out_dim * 2), 
                                    GELU(),
                                    nn.Dropout(args.dropout))

        self.gru_layers = nn.GRU(input_size=out_dim*2, hidden_size=out_dim, num_layers=n_gru_layers,
                                 dropout=args.dropout, bidirectional=True, batch_first=True)
                                 
        self.linear = nn.Sequential(nn.Linear(out_dim * 2, out_dim),
                                    GELU(),
                                    nn.Dropout(args.dropout))


    def forward(self, x, seq_lens):
        """

        :param x: (batch x dim x seq_len) where dim=68
        :param seq_lens: batch
        :return:
        """
        x = self.conv1(x)   # (batch x out_dim//2 x seq_len), where seq_len = [(seq_len + 2*pad - 1) // stride]
        x = self.conv2(x)   # (batch x out_dim*2 x seq_len), where seq_len = [(seq_len + 2*pad - 1) // stride]

        conv_out = torch.transpose(x, 1, 2)    # (batch x seq_len x out_dim*2)

        self.gru_layers.flatten_parameters()
        x, _ = self.gru_layers(conv_out)  # (batch x seq_len x out_dim*2)
        x = x + conv_out
        # pdb.set_trace()
        x = self.linear(x)  # (batch x seq_len x out_dim)
        return x


class VGGModel(nn.Module):
    def __init__(self, args):
        super(VGGModel, self).__init__()
        self.vgg_face = build_vgg(args.vgg_param_dir)
        self.vgg_out_dim = 2622
        out_dim, dropout = args.out_dim, args.dropout
        self.lstm = nn.LSTM(input_size=self.vgg_out_dim, hidden_size=out_dim, num_layers=args.vision_n_lstm, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Sequential(nn.Linear(out_dim*2, out_dim),
                                    GELU(),
                                    nn.Dropout(dropout))

    def forward(self, x):
        """
        x: batch x seq_len x 3 x dim1 x dim2
        """
        batch, seq_len, n_channel, dim1, dim2 = x.shape
        x = x.view(batch*seq_len, n_channel, dim1, dim2)
        x = self.vgg_face(x)    # (batch*seq_len, vgg_out_dim)
        x = x.view(batch, seq_len, self.vgg_out_dim)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)     # (batch, seq_len, out_dim*2)
        x = self.linear(x)
        return x


class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        self.resnet = resnet18(pretrained=args.pretrained_resnet)
        self.resnet_out_dim = 1000
        out_dim, dropout = args.out_dim, args.dropout
        self.gru = nn.GRU(input_size=self.resnet_out_dim, hidden_size=out_dim, num_layers=args.vision_n_gru, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Sequential(nn.Linear(out_dim*2, out_dim),
                                    GELU(),
                                    nn.Dropout(dropout))
    
    def forward(self, x):
        """
        x: batch x seq_len x 3 x dim1 x dim2
        """
        batch, seq_len, n_channel, dim1, dim2 = x.shape
        x = x.view(batch*seq_len, n_channel, dim1, dim2)
        x = self.resnet(x)    # (batch*seq_len, resnet_out_dim)

        x = x.view(batch, seq_len, self.resnet_out_dim)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)     # (batch, seq_len, out_dim*2)
        x = self.linear(x)
        return x


class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        bert_out_dim = 768
        self.linear = nn.Sequential(nn.Linear(bert_out_dim, args.out_dim),
                                    GELU(),
                                    nn.Dropout(args.dropout))    
    
    def forward(self, input_ids, attention_mask):
        # pdb.set_trace()

        outputs = self.bert(input_ids, attention_mask)
        outputs = self.linear(outputs[0])
        return outputs


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        num_layers, model_dim, num_heads, ffw_dim, dropout = \
            args.n_fusion_layers, args.fusion_hid_dim, args.n_attn_heads, args.fusion_ffw_dim, args.transformer_dropout
        self.encoder = Encoder(num_layers, model_dim, num_heads, ffw_dim, dropout)

    def forward(self, x, attn_mask=None):
        x, attentions = self.encoder(x, attn_mask)
        return x, attentions


class Pooler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_state):
        pooled_output = self.dense(hidden_state)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TriModalModel(nn.Module):
    def __init__(self, args):
        super(TriModalModel, self).__init__()

        self.audio_module = AudioModel(args)
        self.text_module = TextModel(args)
        # self.vision_module = VisionModel(args)
        self.vision_module = ResNetModel(args)
        self.fusion_module = FusionModel(args)

        self.num_labels = 6 if args.interview else 5

        self.modal_embeddings = nn.Embedding(3, args.out_dim)
        self.extra_embeddings = nn.Embedding(7, args.out_dim)

        self.LN = nn.LayerNorm(args.out_dim)

        self.classifiers = nn.Sequential(nn.Conv1d(in_channels=args.out_dim*self.num_labels, out_channels=args.out_dim//2*self.num_labels, 
                                                    kernel_size=1, groups=self.num_labels),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=args.out_dim//2*self.num_labels, out_channels=self.num_labels, 
                                                    kernel_size=1, groups=self.num_labels),
                                        nn.Sigmoid())

    def forward(self, audio_feature, audio_len, vision_feature, text_input_ids, 
                text_attn_mask, fusion_attn_mask, extra_token_ids, labels=None):
        audio_x = self.audio_module(audio_feature, audio_len)
        text_x = self.text_module(text_input_ids, text_attn_mask)
        vision_x = self.vision_module(vision_feature)

        # pdb.set_trace()
        # audio_modal_ids = torch.tensor(0, dtype=torch.long, device=audio_x.device).expand(audio_x.shape[0], audio_x.shape[1])
        # text_modal_ids = torch.tensor(1, dtype=torch.long, device=text_x.device).expand(text_x.shape[0], text_x.shape[1])
        # vision_modal_ids = torch.tensor(2, dtype=torch.long, device=vision_x.device).expand(vision_x.shape[0], vision_x.shape[1])
        # audio_emb = self.modal_embeddings(audio_modal_ids)
        # text_emb = self.modal_embeddings(text_modal_ids)
        # vision_emb = self.modal_embeddings(vision_modal_ids)

        # audio_x = audio_x + audio_emb
        # text_x = text_x + text_emb
        # vision_x = vision_x + vision_emb

        cls_emb = self.extra_embeddings(extra_token_ids)
        tmp_sep = torch.tensor(6, dtype=torch.long, device=cls_emb.device).expand(cls_emb.shape[0], 1)
        sep_emb = self.extra_embeddings(tmp_sep)

        fusion_input = torch.cat([cls_emb, audio_x, sep_emb, vision_x, sep_emb, text_x, sep_emb], 1)
        fusion_x = self.LN(fusion_input)
        
        fusion_x, attns = self.fusion_module(fusion_input, fusion_attn_mask)    # batch x seq_len x out_dim
        outputs = (attns, )

        # pdb.set_trace()

        fusion_output = fusion_x[:, :self.num_labels, :].view(fusion_x.shape[0], self.num_labels*fusion_x.shape[2])    # batch x num_labels*out_dim
        fusion_output = fusion_output.unsqueeze(-1)   # batch x num_labels*out_dim x 1

        logits = self.classifiers(fusion_output)
        logits = logits.squeeze(-1)

        outputs = (logits, ) + outputs
        if labels is not None:
            loss = self.compute_MSELoss(labels, logits)
            outputs = (loss,) + outputs

        return outputs

    @staticmethod
    def compute_MSELoss(labels, predictions):
        losses = torch.pow(labels - predictions, 2)
        loss = torch.sum(losses, dim=1)
        loss = torch.mean(loss)
        return loss


class AvgModalModel(nn.Module):
    def __init__(self, args):
        super(AvgModalModel, self).__init__()

        self.audio_module = AudioModel(args)


        self.text_module = TextModel(args)
        # self.vision_module = VisionModel(args)
        self.vision_module = ResNetModel(args)

        self.num_labels = 6 if args.interview else 5
        out_dim = args.out_dim

        self.linear = nn.Sequential(nn.Linear(out_dim * 3, out_dim * self.num_labels),
                                    GELU())
        self.classifiers = nn.Sequential(nn.Conv1d(in_channels=out_dim*self.num_labels, out_channels=self.num_labels, kernel_size=1, groups=self.num_labels),
                                        nn.Sigmoid())

    def forward(self, audio_feature, audio_len, vision_feature, text_input_ids, 
                text_attn_mask, fusion_attn_mask, extra_token_ids, labels=None):
        audio_x = self.audio_module(audio_feature, audio_len)
        text_x = self.text_module(text_input_ids, text_attn_mask)
        vision_x = self.vision_module(vision_feature)

        audio_x = torch.mean(audio_x, 1)
        text_x = torch.mean(text_x, 1)
        vision_x = torch.mean(vision_x, 1)

        x = torch.cat([audio_x, text_x, vision_x], -1)  # batch x out_dim*3

        x = self.linear(x)  # batch x out_dim*num_labels
        x = x.unsqueeze(-1)

        logits = self.classifiers(x)
        logits = logits.squeeze()

        outputs = (logits, )
        if labels is not None:
            loss = self.compute_MSELoss(labels, logits)
            outputs = (loss,) + outputs

        return outputs

    @staticmethod
    def compute_MSELoss(labels, predictions):
        losses = torch.pow(labels - predictions, 2)
        loss = torch.sum(losses, dim=1)
        loss = torch.mean(loss)
        return loss