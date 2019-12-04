import torch
import torch.nn as nn


class FullConnect(nn.Module):
    """docstring for FC"""

    def __init__(self, dim=1000):
        super(FullConnect, self).__init__()
        self.linear = nn.Linear(dim, 5)
        self.sigmod = nn.Sigmoid()

    def forward(self, video_dim):
        after_linear = self.linear(video_dim)
        labels = self.sigmod(after_linear)
        return labels


class MyBiLSTM(nn.Module):
    """docstring for LSTM"""

    def __init__(self, in_dim, hidden_dim, n_layer, n_class, vgg_face, dropout=0.0):
        super(MyBiLSTM, self).__init__()
        self.vgg_16 = vgg_face
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, n_class)
        self.sigmod = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.vgg_16(inputs)
        outputs = outputs.unsqueeze(1)
        out, _ = self.lstm(outputs)  # (image_num, batch_size, feature_dim)
        out = out.squeeze(1)  # (image_num, feature_dim)
        out = self.linear(out)  # (image_num, n_class)
        out, _ = torch.max(out, dim=0)
        out = out.unsqueeze(0)
        out = self.sigmod(out)
        return out


# if __name__ == '__main__':
#     final_sample = torch.zeros([1, 1000])
#     for i in range(10):
#         test_sample = torch.randn(1, 1000)
#         final_sample = torch.cat((final_sample, test_sample), 0)
#     final_sample = final_sample[1:, :]
#     test = MyBiLSTM(1000, 1000, 1, 5)
#     test_sample = final_sample.unsqueeze(1)
#     out = test.forward(test_sample)
