import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        else:
            output = output
        return output


class MLP(nn.Module):
    def __init__(self, in_channel, hidden, num_class, drop_prob):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_channel, hidden, drop_prob, bias=True)
        self.lin2 = Linear(hidden, num_class, drop_prob, bias=True)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.act_fn(self.lin1(x))
        return self.lin2(x)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


class GCN(nn.Module):
    def __init__(self, in_channel, hidden, drop_prob, num_class):
        super(GCN, self).__init__()
        self.Linear1 = Linear(in_channel, hidden, drop_prob, bias=True)
        self.Linear2 = Linear(hidden, num_class, drop_prob, bias=True)

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(torch.matmul(adj, x)))
        h = self.Linear2(torch.matmul(adj, x))
        return torch.log_softmax(h, dim=-1)

    def reset_parameters(self):
        self.Linear1.reset_parameters()
        self.Linear2.reset_parameters()
