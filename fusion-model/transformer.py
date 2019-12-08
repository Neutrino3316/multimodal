import torch
import torch.nn as nn
from decode import decoder
from encode import encoder


class transformer(nn.Module):
    """docstring for transformer"""

    def __init__(self, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(transformer, self).__init__()
        self.CLS = nn.Parameter(torch.FloatTensor(6, model_dim))
        self.SEP = nn.Parameter(torch.FloatTensor(1, model_dim))
        self.enc = encoder(num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.dec = decoder(num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, 5, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Audio, Text, Vision):
        input_modal = torch.cat((self.CLS, Audio, self.SEP, Text, self.SEP, Vision, self.SEP), 0)
        input_modal = torch.unsqueeze(input_modal, 0)
        output, enc_self_attn = self.enc(input_modal)
        output, dec_self_attn, dec_attn = self.dec(input_modal, output)
        output = self.linear(output)
        print(output.shape)
        output = torch.mean(output, dim=1)
        output = self.softmax(output)
        return output, enc_self_attn, dec_self_attn, dec_attn


if __name__ == '__main__':
    test_transformer = transformer()
    audio = torch.FloatTensor(15, 512)
    text = torch.FloatTensor(15, 512)
    vision = torch.FloatTensor(15, 512)
    output, enc_self_attn, dec_self_attn, dec_attn = test_transformer.forward(audio, text, vision)
    print(output.shape)
