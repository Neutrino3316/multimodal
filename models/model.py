import torch
import torch.nn as nn

from Vgg_face import build_vgg


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

        x_conv = torch.transpose(x, 1, 2)    # (batch x seq_len x out_dim*2)

        x, _ = self.gru_layers(x_conv)  # (batch x seq_len x out_dim*2)
        x = x + x_conv
        # pdb.set_trace()
        x = self.linear(x)  # (batch x seq_len x out_dim)
        return x


class VisionModel(nn.Module):
    def __init__(self, args)
        super(VisionModel, self).__init__()
        self.vgg_face = build_vgg(args.vgg_param_dir)
        out_dim, dropout = args.out_dim, args.dropout
        self.lstm = nn.LSTM(input_size=2622, hidden_size=out_dim, num_layers=args.vision_n_gru, dropout=dropout, bidirectional=True)
        self.linear = nn.Sequential([nn.Linear(out_dim*2, out_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout)])

    def forward(self, x_conv, batch_size):
        """
        param x_conv: (seq_len*batch_size, 3, 224, 224)
        param batch_size: > 1
        """
        outputs = self.vgg_16(inputs)  # (batch_size*seq_len, feature_dim)
        outputs = outputs.view(batch_size, -1, 2622)  # (batch_size, seq_len, feature_dim)
        outputs = torch.transpose(outputs, 0, 1)  # (seq_len, batch_size, feature_dim)
        out, _ = self.lstm(outputs)  # (seq_len, batch_size, feature_dim)
        out = self.linear(out) # (seq_len, batch_size, n_class)
        out, _ = torch.max(out, dim=0) # (batch_size, n_class)
        return out

class TriModalModel(nn.Module):
    def __init__(self, args):
        super(TriModalModel, self).__init__()

        self.audio_module = AudioModel(args)
        self.text_module = TextModel(args)
        self.vision_module = VisionModel(args)
        self.fusion_module = Transformer(args)

        self.classifier = nn.Sequential()

    def forward(self, audio_feature, audio_len, vision_feature, text_input_ids, text_attn_mask, y):
        audio_x = self.audio_module(audio_feature, audio_len)
        text_x = self.text_module(text_input_ids, text_attn_mask)
        vision_x = self.vision_module(vision_feature)
        
        fusion_x = self.fusion_module(audio_x, text_x, vision_x, text_attn_mask)