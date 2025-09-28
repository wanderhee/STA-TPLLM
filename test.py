import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGCN_r import make_model, ASTGCN_block
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, \
    predict_and_save_results_mstgcn, scaled_Laplacian, cheb_polynomial
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)

# TensorDataset 使用索引调用数据 dataloader
# 例train_loader里面可以取input和label
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)


# 构建邻接矩阵
adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

x = torch.rand(32, 307, 1, 12)
x = x.to(DEVICE)
print(x.shape) # 32,307,1,12

# net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
#                  num_for_predict, len_input, num_of_vertices)
#
# output = net(x)
#
# print(output.shape)

# grucell = nn.GRU(input_size=307, hidden_size=307, num_layers=2, device=DEVICE)
#
# x = torch.squeeze(x)
# x = x.permute(2, 0, 1)
#
# # gru输入：12，32，307
# out, hn = grucell(x)
#
# print(x.shape) # 12,32,307
# print(out.shape) # 12,32,307
# print(hn.shape) # 2,32,307
#
# out = torch.unsqueeze(out, 3) # 12,32,307,1
# out = out.permute(1, 2, 3, 0)
# print(out.shape) # 32,307,1,12