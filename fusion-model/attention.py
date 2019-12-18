import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ScaleDotProductAttention(nn.Module):
    """docstring for ScaleDotProductAttention"""

    def __init__(self, dropout=0.0):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        其中q,k,v大小一致，都是[batch_size, file_size, word_embedding_size]
        返回上下文张量和attention分数
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
    """docstring for MultiHeadAttention"""

    def __init__(self, model_dim=512, num_head=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d = model_dim // num_head
        self.num_head = num_head
        # [batch_size, file_size, self.num_head*self.d]
        self.linear_k = nn.Linear(model_dim, self.num_head * self.d)
        self.linear_q = nn.Linear(model_dim, self.num_head * self.d)
        self.linear_v = nn.Linear(model_dim, self.num_head * self.d)

        self.dotAttention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

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
        output = self.linear_final(context)
        output = self.dropout(output)
        # residual需要吗 答：有需要，可以解决梯度消失问题
        residual = query
        output = self.layer_norm(output + residual)

        return output, attention


class PositionwiseFeedForward(nn.Module):
    """docstring for PositionwiseFeedForward"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(d_in, d_hid, 1)
        self.w2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(residual + output)
        return output


if __name__ == '__main__':
    test = MultiHeadAttention()
    a = torch.FloatTensor(1, 3, 512)
    b = torch.FloatTensor(1, 3, 512)
    c = torch.FloatTensor(1, 3, 512)

    test.forward(a, b, c)
    test_tensor = torch.FloatTensor([5.0, 5.0, 5.0, 5.0, 5.0])
    test_tensor2 = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0])
    test_tensor3 = torch.abs(test_tensor2 - test_tensor)
    print(test_tensor.shape[0])
    plt.plot(range(test_tensor.shape[0]), test_tensor, ls="-", lw=2, label="plot figure")
    plt.show()
    print(test_tensor3 / 5)
    test_tensor4 = torch.LongTensor([5])
    if test_tensor4 < 6:
        print('YES')
