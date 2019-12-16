import torch
import torch.nn as nn

from Vgg_face import build_vgg

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)


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
    def __init__(self, args)
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
    def __init__(self, args)
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


class PositionwiseFeedForward(nn.Module):
    """docstring for PositionwiseFeedForward"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)

    def forward(self, x):
        output = self.w2(F.relu(self.w1(x)))
        return output


class ScaleDotProductAttention(nn.Module):
    """docstring for ScaleDotProductAttention"""

    def __init__(self, dropout=0.0):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [batch_size, file_size, word_embedding_size]
        """
        dk = q.shape[2]
        attention = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(dk)  # [b, n, m]*[b, m, n] batch matrix-matrix product
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.1)
        super(MultiHeadAttention, self).__init__()
        self.d = model_dim // num_head
        self.num_head = num_head
        # [batch_size, file_size, self.num_head*self.d]
        self.linear_k = nn.Linear(model_dim, self.num_head * self.d)
        self.linear_q = nn.Linear(model_dim, self.num_head * self.d)
        self.linear_v = nn.Linear(model_dim, self.num_head * self.d)

        self.dotAttention = ScaleDotProductAttention(dropout)

    def forward(self, key, value, query, mask=None):
        d = self.d
        num_head = self.num_head
        batch_size = key.shape[0]
        # linear projection
        k = self.linear_k(key)
        q = self.linear_q(query)
        v = self.linear_v(value)

        # tensor transform
        k = k.view(batch_size * num_head, -1, d)
        q = q.view(batch_size * num_head, -1, d)
        v = v.view(batch_size * num_head, -1, d)

        # self attention
        context, attention = self.dotAttention(q, k, v, mask)
        context = context.view(batch_size, -1, num_head * d)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, ffw_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.Attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.LN1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.Feed_forward = PositionwiseFeedForward(model_dim, ffw_dim, dropout)
        self.LN2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        context, attention = self.Attention(x, x, x, attn_mask)
        x = self.LN1(x + self.dropout1(context))
        output = self.Feed_forward(x)
        output = self.LN2(x + self.dropout2(output))
        return output, attention


class Transformer_Encoder(nn.Module):
    def __init__(self, args):
        super(Transformer_Encoder, self).__init__()
        num_layers, model_dim, num_heads, ffw_dim, dropout = 
            args.n_fusion_layers, args.fusion_hid_dim, args.n_attn_heads, args.fusion_ffw_dim, args.transformer_dropout
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffw_dim, dropout) 
                                            for _ in range(num_layers)])

    def forward(self, x, attn_mask=None):
        attentions = ()
        for enc in self.encoder_layers:
            x, attention = enc(x, attn_mask)
            attentions = attentions + (attention, )
        return x, attentions


class TriModalModel(nn.Module):
    def __init__(self, args):
        super(TriModalModel, self).__init__()

        self.audio_module = AudioModel(args)
        self.text_module = TextModel(args)
        self.vision_module = VisionModel(args)
        self.fusion_module = Transformer_Encoder(args)

        num_labels = if args.interview else 5
        self.CLS = nn.Parameter(torch.FloatTensor(num_labels, args.out_dim))
        self.SEP = nn.Parameter(torch.FloatTensor(1, args.out_dim))

        self.classifier = nn.Sequential()

    def forward(self, audio_feature, audio_len, vision_feature, text_input_ids, text_attn_mask, fusion_attn_mask, y):
        audio_x = self.audio_module(audio_feature, audio_len)
        text_x = self.text_module(text_input_ids, text_attn_mask)
        vision_x = self.vision_module(vision_feature)

        fusion_input = torch.cat([self.CLS, audio_x, self.SEP, vision_x, self.SEP, text_x], 1)
        
        fusion_x = self.fusion_module(fusion_input, fusion_attn_mask)