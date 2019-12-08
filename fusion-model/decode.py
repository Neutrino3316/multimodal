import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from attention import MultiHeadAttention, ScaleDotProductAttention, PositionwiseFeedForward


class decoderLayer(nn.Module):
    """docstring for decoderLayer"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(decoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_input, enc_output):
        dec_output, self_attention = self.attention(dec_input, dec_input, dec_input)
        dec_output, enc_attention = self.attention(enc_output, enc_output, dec_output)
        output = self.feed_forward(dec_output)
        return output, self_attention, enc_attention


class decoder(nn.Module):
    """ docstring for decoder"""

    def __init__(self, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(decoder, self).__init__()
        self.decoder_layer = nn.ModuleList(
            [decoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    def forward(self, inputs, encode_output, context_attn_mask=None):
        output = inputs
        self_attention = []
        context_attention = []
        for dec in self.decoder_layer:
            output, self_attn, context_attn = dec(output, encode_output)
            self_attention.append(self_attn)
            context_attention.append(context_attn)
        return output, self_attention, context_attention


if __name__ == '__main__':
    inputs = torch.FloatTensor(1, 54, 512)
    encode_output = torch.FloatTensor(1, 54, 512)
    print(inputs.shape)
    test_decoder = decoder()
    output, self_attn, context_attn = test_decoder.forward(inputs, encode_output)
    print(output.shape, len(self_attn))
