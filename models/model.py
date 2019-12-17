import torch
import torch.nn as nn

from Vgg_face import build_vgg
from transformer import Encoder

from transformers import BertModel


class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        input_dim, out_dim, n_gru_layers = 68, args.out_dim, args.audio_n_gru
        n_cls = 6 if args.interview else 5
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=out_dim//2, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=output_dim//2, out_channels=out_dim*2, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.gru_layers = nn.GRU(input_size=out_dim*2, hidden_size=out_dim, num_layers=n_gru_layers,
                                 dropout=args.dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Sequential([nn.Linear(out_dim * 2, out_dim),
                                    nn.ReLU(),
                                    nn.Dropout(args.dropout)])


    def forward(self, x, seq_lens):
        """

        :param x: (batch x dim x seq_len) where dim=68
        :param seq_lens: batch
        :return:
        """
        x = self.conv1(x)   # (batch x out_dim//2 x seq_len), where seq_len = [(seq_len + 2*pad - 1) // stride]
        x = self.relu1(x)
        x = self.conv2(x)   # (batch x out_dim x seq_len), where seq_len = [(seq_len + 2*pad - 1) // stride]
        x = self.relu2(x)

        conv_out = torch.transpose(x, 1, 2)    # (batch x seq_len x out_dim*2)

        x, _ = self.gru_layers(x_conv)  # (batch x seq_len x out_dim*2)
        x = x + conv_out
        # pdb.set_trace()
        x = self.linear(x)  # (batch x seq_len x out_dim)
        return x


class VisionModel(nn.Module):
    def __init__(self, args):
        super(VisionModel, self).__init__()
        self.vgg_face = build_vgg(args.vgg_param_dir)
        self.vgg_out_dim = 2622
        out_dim, dropout = args.out_dim, args.dropout
        self.lstm = nn.LSTM(input_size=self.vgg_out_dim, hidden_size=out_dim, num_layers=args.vision_n_gru, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Sequential([nn.Linear(out_dim*2, out_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout)])

    def forward(self, x):
        """
        x: batch x seq_len x 3 x dim1 x dim2
        """
        batch, seq_len, n_channel, dim1, dim2 = x.shape
        x = x.view(batch*seq_len, n_channel, dim1, dim2)
        x = self.vgg_face(x)    # (batch*seq_len, vgg_out_dim)
        x = x.view(batch, seq_len, self.vgg_out_dim)
        x, _ = self.lstm(x)     # (batch, seq_len, out_dim*2)
        x = self.linear(x)
        return x


class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        bert_out_dim = 768
        self.linear = nn.Sequential([nn.Linear(bert_out_dim, out_dim),
                                        nn.ReLU(),
                                        nn.Dropout(args.dropout)])    
    
    def forward(self, input_ids, attention_mask):
        _, outputs, _ = self.bert(input_ids, attention_mask)
        outputs = self.linear(outputs)
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
    def __init__(self, args):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(args.out_dim, args.out_dim // 2)
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
        self.vision_module = VisionModel(args)
        self.fusion_module = FusionModel(args)

        self.num_labels = 6 if args.interview else 5
        self.CLS = nn.Parameter(torch.FloatTensor(num_labels, args.out_dim))
        self.SEP = nn.Parameter(torch.FloatTensor(1, args.out_dim))

        self.classifiers = nn.ModuleList([nn.Sequential([
                                            Pooler(args),
                                            nn.Dropout(args.dropout),
                                            nn.Linear(args.out_dim//2, 1),
                                            nn.Sigmoid()])
                                         for _ in range(self.num_labels)])

    def forward(self, audio_feature, audio_len, vision_feature, text_input_ids, 
                text_attn_mask, fusion_attn_mask, labels=None):
        audio_x = self.audio_module(audio_feature, audio_len)
        text_x = self.text_module(text_input_ids, text_attn_mask)
        vision_x = self.vision_module(vision_feature)

        fusion_input = torch.cat([self.CLS, audio_x, self.SEP, vision_x, self.SEP, text_x], 1)        
        fusion_x, _ = self.fusion_module(fusion_input, fusion_attn_mask)

        logits = ()
        for i, clf in enumerate(self.classifiers):
            hidden_state = fusion_x[:, i]
            logit = clf(hidden_state)
            logits = logits + (logit, )
        logits = torch.stack(logits)

        outputs = (logits, )
        if labels is not None:
            
            # TODO
        return outputs
    