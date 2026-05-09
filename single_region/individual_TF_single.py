import torch.nn as nn
# from transformer.decoder import Decoder
# from transformer.multihead_attention import MultiHeadAttention
# from transformer.positional_encoding import PositionalEncoding
# from transformer.pointerwise_feedforward import PointerwiseFeedforward
# from transformer.encoder_decoder import EncoderDecoder
# from transformer.encoder import Encoder
# from transformer.encoder_layer import EncoderLayer
# from transformer.decoder_layer import DecoderLayer
import copy
import torch.nn.functional as F
import torch
import math


# class IndividualTF(nn.Module):
#     def __init__(self):
#         super(IndividualTF, self).__init__()
#         "Helper: Construct a model from hyperparameters."
#         c = copy.deepcopy
#         self.conv1=nn.Conv2d(1, 32, (1, 3), stride=(1, 2))
#         self.conv2=nn.Conv2d(32, 64, (1, 3), stride=(1, 2))
#         self.conv3=nn.Conv2d(64, 128, (1, 3), stride=(1, 2))
#         self.conv4 = nn.Conv2d(128, 128, (1, 3), stride=(1, 2))
#         self.pool = nn.AvgPool2d((1,42), stride=1)
#         self.linear1 = nn.Linear(128, 64)
#         self.linear2 = nn.Linear(64, 64)
#         self.classifier=nn.Sequential(
#             nn.Linear(64, 64),
#             # nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Linear(64, 2),
#         )
#
#     def forward(self,input):
#         # self.
#         # print(input.shape)
#         print(input.shape)
#         out=self.conv1(input)
#         print(out.shape)
#
#         out=self.conv2(out)
#         print(out.shape)
#         out = self.conv3(out)
#         print(out.shape)
#         out = self.conv4(out)
#         print(out.shape)
#         out=self.pool(out)
#         print(out.shape)
#         # print(out.shape)
#         # exit(0)
#         out=torch.squeeze(out)
#         print(out.shape)
#         out=self.linear1(out)
#         print(out.shape)
#         out=self.linear2(out)
#         print(out.shape)
#         out=self.classifier(out)
#         print(out.shape)
#         exit()
#         return out
class IndividualTF(nn.Module):
    def __init__(self):
        super(IndividualTF, self).__init__()

        # Flatten 输入: (16, 1, 1, 701) -> (16, 701)
        # self.flatten = nn.Flatten()
        self.softmax=nn.Softmax(dim=1)

        # 三个全连接层
        self.fc1 = nn.Linear(701, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, input,pred=True):
        # print(input.shape)  # (16, 1, 1, 701)

        # out = self.flatten(input)
        # print(out.shape)  # (16, 701)

        out = self.fc1(input)
        # print(out.shape)  # (16, 256)
        out = self.fc2(out)
        # print(out.shape)  # (16, 128)
        out = self.fc3(out)
        # print(out.shape)  # (16, 64)

        out = self.classifier(out)
        if pred:
            out=self.softmax(out)
        # print(out)
        # print(out.shape)
        # probs = F.softmax(out, dim=1)#.cpu().numpy()
        # print(probs.shape)
        # print(out.shape)  # (16, 2)
        # exit()

        return out
# class LinearEmbedding_sp(nn.Module):
#     def __init__(self, inp_size,d_model):
#         super(LinearEmbedding_sp, self).__init__()
#         self.lut = nn.Linear(inp_size, d_model)
#         self.d_model = d_model
#
#     def forward(self, x):
#         return self.lut(x)
#
# class LinearEmbedding(nn.Module):
#     def __init__(self, inp_size,d_model):
#         super(LinearEmbedding, self).__init__()
#         # lut => lookup table
#         self.lut = nn.Linear(inp_size, d_model)
#         self.d_model = d_model
#
#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)
#
#
# class Generator(nn.Module):
#     "Define standard linear + softmax generation step."
#
#     def __init__(self, d_model, out_size):
#         super(Generator, self).__init__()
#         self.proj=nn.Linear(d_model,out_size)
#
#     def forward(self, x):
#         return self.proj(x)
