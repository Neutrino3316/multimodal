import torch
import torch.nn as nn
from attention import MultiHeadAttention
from attention import PositionwiseFeedForward


class encodeLayer(nn.Module):
    """docstring for encodeLayer"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(encodeLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


class encoder(nn.Module):
    """docstring for encoder"""

    def __init__(self, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [encodeLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    def forward(self, inputs):
        output = inputs
        attentions = []
        for enc in self.encoder_layers:
            output, attention = enc(output)
            attentions.append(attention)
        return output, attentions


if __name__ == '__main__':
    inputs = torch.FloatTensor(1, 54, 512)
    print(inputs.shape)
    test_encoder = encoder()
    output, attn = test_encoder.forward(inputs)
    print(output.shape, len(attn))
