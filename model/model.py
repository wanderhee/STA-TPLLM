# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
# from lib.utils import scaled_Laplacian, cheb_polynomial

from transformers import LlamaModel, LlamaConfig
# from transformers import BertTokenizer, BertModel
# from einops import rearrange
from peft import prepare_model_for_kbit_training, get_peft_config, get_peft_model, LoraConfig, TaskType


class TimeEmb(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, dropout = 0.2):
        super(TimeEmb, self).__init__()
        self.gru = nn.GRU(input_size=num_of_vertices, hidden_size=num_of_vertices, num_layers=2, device=DEVICE)
        self.Wq1 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_timesteps).to(DEVICE))  # NT
        self.Wk1 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_timesteps).to(DEVICE))  # NT
        self.Wv1 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))  # TT
        self.d1 = num_of_timesteps

        # self.V1 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))  # TT 加这句
        # self.b1 = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))  # 1TT 加这句

        self.dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(num_of_timesteps, num_of_timesteps)
        self.ReLU = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # nn.GRU会自动初始化
        init.kaiming_uniform_(self.Wq1)
        init.kaiming_uniform_(self.Wk1)
        init.kaiming_uniform_(self.Wv1)

        init.kaiming_uniform_(self.FC.weight)
        if self.FC.bias is not None:
            init.zeros_(self.FC.bias)

    def forward(self, x):
        '''
        :param x: B, N, F, T
        :return: B, N, F(1), T
        '''

        x = x[:, :, -1, :]  # B, N, T
        x_v = x.permute(0, 2, 1)  # B, T, N
        gru_in = x.permute(2, 0, 1)  # T, B, N
        gru_out, _ = self.gru(gru_in)  # T, B, N  可以试试不要GRU
        gru_out = gru_out.permute(1, 0, 2)  # B, T, N

        Q1 = torch.matmul(gru_out, self.Wq1)  # BTN * NT = BTT
        K1 = torch.matmul(gru_out, self.Wk1).permute(0, 2, 1)  # (BTN * NT)^T = BTT
        V1 = torch.matmul(self.Wv1, x_v)  # TT * BTN = BTN

        product = torch.matmul(Q1, K1)  # BTT * BTT = BTT
        T_Att = torch.div(product, math.sqrt(self.d1))  # QK/根号d BTT

        # T_Att = torch.matmul(self.V1, torch.sigmoid(T_Att + self.b1))  # TT x sig(BTT+1TT) 加这句

        T_Att = F.softmax(T_Att, dim=1)
        T_Att = self.dropout(T_Att)

        out = torch.matmul(T_Att, V1)  # BTT * BTN = BTN
        out = out.permute(0, 2, 1)  # BNT

        # 线性层
        out = self.FC(out)
        out = self.ReLU(out)

        out = torch.unsqueeze(out, 2) # B, N, F, T

        return out


class Spatial_Attention(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, dropout = 0.2):
        super(Spatial_Attention, self).__init__()
        self.Wq2 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_vertices).to(DEVICE))  # TN
        self.Wk2 = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_vertices).to(DEVICE))  # TN
        self.d2 = num_of_vertices
        # self.V2 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))  # NN 这个可能不需要
        # self.b2 = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))  # 1NN
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.Wq2)
        init.kaiming_uniform_(self.Wk2)

        # init.xavier_uniform_(self.V2)
        # init.xavier_uniform_(self.b2)

    def forward(self, x):
        '''
        :param x: B, N, F, T
        :return: B, N, N
        '''

        x_in = x[:, :, -1, :]  # B, N, T
        Q2 = torch.matmul(x_in, self.Wq2)  # BNT * TN = BNN
        K2 = torch.matmul(x_in, self.Wk2).permute(0, 2, 1)  # (BNT * TN)^T = BNN

        product = torch.matmul(Q2, K2)  # BNN * BNN = BNN
        # S_Att = F.softmax(torch.div(product, math.sqrt(self.d2)), dim=1)  # QK/根号d BNN

        S_Att = torch.div(product, math.sqrt(self.d2))
        # S_Att = torch.matmul(self.V2, torch.sigmoid(S_Att + self.b2))  # !new!
        S_Att = F.softmax(S_Att, dim=1)

        S_Att = self.dropout(S_Att)  # BNN

        return S_Att


class Graph_Conv_with_SAtt(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, adj, in_channels, out_channels, bias=True):
        super(Graph_Conv_with_SAtt, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps
        self.adj = adj.to(DEVICE)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(DEVICE))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).to(DEVICE))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, S_Att):
        '''
        :param x: B, N, in_channels(1), T
        :return: B, N, out_channels, T
        '''

        outputs = []

        for time_step in range(self.num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            L_with_SA = torch.matmul(self.adj, S_Att)  # N N * B N N = B N N

            support = torch.matmul(graph_signal, self.weight)  # B N F_in * F_in F_out = B N F_out

            output = torch.matmul(L_with_SA.permute(0, 2, 1), support)  # B N N * B N F_out = B N F_out  这里是否需要permute？

            if self.bias is not None:
                output = output + self.bias

            outputs.append(output.unsqueeze(-1))  # B N F_out 1

        return F.relu(torch.cat(outputs, dim=-1))  # B N F_out T


class LLaMA3_Lora_block_no_emb(nn.Module):

    def __init__(self, DEVICE, num_for_predict, d_model):
        super(LLaMA3_Lora_block_no_emb, self).__init__()
        # 修改为LLaMA3的本地路径
        self.pretrained_model_path = '/llama3/model/'  # LLaMA3模型路径
        
        # 加载LLaMA3模型配置和权重
        config = LlamaConfig.from_pretrained(self.pretrained_model_path)
        self.llama3 = LlamaModel.from_pretrained(
            self.pretrained_model_path, 
            config=config,
            torch_dtype=torch.float16,  
            device_map="auto"
        )
        
        # 配置LoRA参数
        self.peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1,  # 降低dropout以适应LLaMA3
            target_modules=["q_proj", "v_proj"],  # LLaMA3的注意力模块名称
            bias="none"
        )
        
        # 准备模型并进行LoRA适配
        self.llama3 = prepare_model_for_kbit_training(self.llama3)
        self.lora_llama3 = get_peft_model(self.llama3, self.peft_config)
        
        self.out_layer = nn.Linear(config.hidden_size, num_for_predict)

        for layer in (self.lora_llama3, self.out_layer):
            layer.to(DEVICE)

    def forward(self, x):
        '''
        :param x: B, N, d_model
        :return: B, N, T
        '''
        # LLaMA3的输入处理
        outputs = self.lora_llama3(inputs_embeds=x, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (B, N, hidden_size)
        
        output = self.out_layer(hidden_states)  # (B, N, T)

        return output


class STATPLLM(nn.Module):

    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, adj, in_channels, nb_filter, time_strides, d_model, num_for_predict):
        super(STATPLLM, self).__init__()
        self.TE = TimeEmb(DEVICE, num_of_vertices, num_of_timesteps)
        self.TE_conv = nn.Conv2d(in_channels, nb_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.SA = Spatial_Attention(DEVICE, num_of_vertices, num_of_timesteps)
        self.SE = Graph_Conv_with_SAtt(DEVICE, num_of_vertices, num_of_timesteps, adj, in_channels=in_channels,
                                       out_channels=nb_filter)  # in_channels=1 nb_filter=64
        self.Wv2 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))  # NN new
        self.b1 = nn.Parameter(torch.FloatTensor(1, nb_filter, num_of_vertices, num_of_timesteps).to(DEVICE))  # 1 64 N T new
        self.b2 = nn.Parameter(torch.FloatTensor(1, nb_filter, num_of_vertices, num_of_timesteps).to(DEVICE))  # 1 64 N T new
        self.residual_conv = nn.Conv2d(in_channels, nb_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.GELU = nn.GELU()
        self.leaky_relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(nb_filter)  # 需要将channel放到最后一个维度上
        self.input_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, 1))
        # 修改类名
        self.llama3_lora = LLaMA3_Lora_block_no_emb(DEVICE, num_for_predict, d_model)
        self.reset_parameters()

        for layer in (self.TE, self.TE_conv, self.SA, self.SE, self.residual_conv, self.GELU, self.leaky_relu, self.ln, self.input_conv, self.llama3_lora):
            layer.to(DEVICE)

    def reset_parameters(self):
        init.kaiming_uniform_(self.Wv2)
        init.kaiming_uniform_(self.b1)
        init.kaiming_uniform_(self.b2)
        # self.Wv2.fill_(0.9)
        

    def forward(self, x):
        '''
        :param x: B, N, in_channels, T
        :return: B, N, T
        '''

        # Time Embedding
        x_TE = self.TE(x)  # B N 1 T -> B N 1 T
        x_TE = F.relu(self.TE_conv(x_TE.permute(0, 2, 1, 3)) + self.b1)  # B 64 N T relub1new!

        # Graph Embedding
        SA = self.SA(x)  # B N 1 T -> B N N

        # x_s = self.leaky_relu(torch.matmul(self.Wv2, x.permute(0, 2, 1, 3)))  # NN * B1NT = B1NT NEW
        x_s = torch.matmul(self.Wv2, x.permute(0, 2, 1, 3))  # NN * B1NT = B1NT NEW
        x_s = x_s.permute(0, 2, 1, 3)  # BN1T

        x_SE = self.SE(x_s, SA)  # B N 64 T
        x_SE = F.relu(x_SE.permute(0, 2, 1, 3) + self.b2)  # B 64 N T relub2new!
        # 这里可以加一个GRGCAN里的time_conv

        # Residual
        x_r = self.residual_conv(x.permute(0, 2, 1, 3))  # B 1 N T -> B F(64) N T    1x1卷积
        
        #x_in = self.GELU(x_TE + x_SE + x_r).permute(0, 3, 2, 1)
        x_in = self.GELU(x_TE + x_SE).permute(0, 3, 2, 1)  # B F(64) N T -> B T N F(64)
        x_in = self.ln(x_in).permute(0, 2, 3, 1)  # B T N F -> B N F T
        x_in = self.input_conv(x_in.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)  # BNFT->BTNF->B,d_model,N->B,N,d_model 相当于in_layer

        # 调用
        output = F.relu(self.llama3_lora(x_in))  # B N T

        return output


def make_STATPLLM_model(DEVICE, len_input, time_strides, in_channels, nb_chev_filter, adj_mx, num_of_vertices, d_model, num_for_predict):

    model = STATPLLM(DEVICE=DEVICE,
                   num_of_vertices=num_of_vertices,
                   num_of_timesteps=len_input,
                   adj=adj_mx,
                   in_channels=in_channels,
                   nb_filter=nb_chev_filter,
                   time_strides=time_strides,
                   d_model=d_model,
                   num_for_predict=num_for_predict
                   )

    return model