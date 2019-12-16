import torch
import torch.nn as nn
from Vgg_face import vgg_face_dag


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
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, n_class)
        self.sigmod = nn.Sigmoid()

    def forward(self, inputs, batch_size):
        outputs = self.vgg_16(inputs)  # (batch_size*seq_len, feature_dim)
        # print(outputs.shape)
        outputs = outputs.view(batch_size, -1, 2622)  # (batch_size, seq_len, feature_dim)
        outputs = torch.transpose(outputs, 0, 1)  # (seq_len, batch_size, feature_dim)
        # print(outputs.shape)
        out, _ = self.lstm(outputs)  # (seq_len, batch_size, feature_dim)
        # print(out.shape)
        # out = out.squeeze(1)  # (seq_len, feature_dim)
        out = self.linear(out)  # (seq_len, batch_size, n_class)
        # print(out.shape)
        out, _ = torch.max(out, dim=0)  # (batch_size, n_class)
        # print(out.shape)
        out = self.sigmod(out)  # (batch_size, n_class)
        # print(out.shape)
        return out


if __name__ == '__main__':
    final_sample = torch.zeros([1, 3, 224, 224])
    for i in range(20):
        test_sample = torch.randn(1, 3, 224, 224)
        final_sample = torch.cat((final_sample, test_sample), 0)
    final_sample = final_sample[1:, :]
    vgg = vgg_face_dag('./data/vgg_face_dag.pth')
    test = MyBiLSTM(2622, 2622, 1, 5, vgg)
    print(test_sample.shape)
    out = test.forward(final_sample, 2)

    test_mean = torch.randn(3, 5)
    print(test_mean.mean(dim=0))
