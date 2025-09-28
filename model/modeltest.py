import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from model import make_STATPLLM_model

DEVICE = torch.device('cuda:0')
adj_mx = torch.rand(307, 307, dtype=torch.float32)
x = torch.rand(32, 307, 1, 12).to(DEVICE)

net = make_STATPLLM_model(DEVICE, len_input=12, time_strides=1, in_channels=1, nb_chev_filter=64, adj_mx=adj_mx, num_of_vertices=307, d_model=768, num_for_predict=12)

total_size = 0
trainable_size = 0

for name, param in net.named_parameters():

    print(name, ":", param.shape, "---", "requires_grad :", param.requires_grad)

    if param.requires_grad:
        lst = list(param.shape)
        size = 1
        for item in lst:
            size *= item
        trainable_size += size

    if True:
        lst2 = list(param.shape)
        size2 = 1
        for item in lst2:
            size2 *= item
        total_size += size2

print("all_paras :", total_size)
print("trainable_paras :", trainable_size)
print("trainable_paras_ratio :", trainable_size / total_size)

y = net(x)

print(y.shape)
print(y)