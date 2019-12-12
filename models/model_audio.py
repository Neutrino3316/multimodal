import torch
import torch.nn as nn

import pdb

class audio_model(nn.Module):
    def __init__(self, args):
        super(audio_model, self).__init__()
        input_dim, output_dim, n_gru_layers = 68, args.out_dim, args.n_gru_layers
        n_cls = 6 if args.interview else 5
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=output_dim//4, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=output_dim//4, out_channels=output_dim, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.gru_layers = nn.GRU(input_size=output_dim, hidden_size=output_dim//2, num_layers=n_gru_layers,
                                 bidirectional=True, batch_first=True)
        self.classifiers = nn.Sequential(nn.Linear(output_dim, output_dim//4),
                                          nn.Tanh(),
                                          nn.Linear(output_dim//4, n_cls),
                                          nn.Sigmoid())

    def forward(self, x, seq_lens, y=None):
        """

        :param x: (batch x dim x seq_len) where dim=68
        :param seq_lens: batch
        :return:
        """
        x = self.conv1(x)   # (batch x output_dim//4 x seq_len), where seq_len = [(seq_len + 2*pad - 1) // stride]
        x = self.relu1(x)
        x = self.conv2(x)   # (batch x output_dim x seq_len), where seq_len = [(seq_len + 2*pad - 1) // stride]
        x = self.relu2(x)

        x = torch.transpose(x, 1, 2)    # (batch x seq_len x output_dim)

        x, _ = self.gru_layers(x)  # (batch x seq_len x output_dim)
        # pdb.set_trace()
        x = torch.sum(x, dim=1)    # (batch x output_dim)
        x = x / seq_lens.unsqueeze(1)

        probs = self.classifiers(x)    # (batch x n_cls)

        if y is not None:
            # return self.compute_MSELoss(y, probs)
            return self.compute_HuberLoss(y, probs)
        else:
            return probs

    def compute_MSELoss(self, labels, predictions):
        losses = torch.pow(labels - predictions, 2)
        loss = torch.sum(losses, dim=1)
        loss = torch.mean(loss)
        return loss

    def compute_HuberLoss(self, labels, predictions):
        delta = 0.1
        error = torch.abs(labels - predictions)
        # pdb.set_trace()
        larger_than_delta_mask = (error > delta).float()
        error_1 = (error * delta - 0.5 * delta ** 2) * larger_than_delta_mask
        error_2 = 0.5 * torch.pow(error, 2) * (1 - larger_than_delta_mask)
        error = error_1 + error_2
        loss = torch.sum(error, dim=1)
        loss = torch.mean(loss)
        return loss