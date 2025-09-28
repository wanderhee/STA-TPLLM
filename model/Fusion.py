import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from fasterkan import FasterKAN as KAN
from transformers import LlamaModel, LlamaConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType


class KANFeatureFusion(nn.Module):
    """
    KAN-based feature fusion module for spatio-temporal features
    """
    def __init__(self, DEVICE, temporal_dim, spatial_dim, fusion_dim, hidden_dim=64):
        super(KANFeatureFusion, self).__init__()
        
        # 输入维度: temporal_dim + spatial_dim
        # 输出维度: fusion_dim
        self.kan_fusion = KAN([temporal_dim + spatial_dim, hidden_dim, fusion_dim])
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.kan_fusion.to(DEVICE)
        self.layer_norm.to(DEVICE)

    def forward(self, temporal_features, spatial_features):
        """
        Fuse temporal and spatial features using KAN
        
        Args:
            temporal_features: (B, N, temporal_dim) - temporal features from GRU/CNN
            spatial_features: (B, N, spatial_dim) - spatial features from GCN
            
        Returns:
            fused_features: (B, N, fusion_dim) - fused spatio-temporal features
        """
        B, N, _ = temporal_features.shape
        
        # 时空特征融合，并调整为KAN适配的格式 
        combined_features = torch.cat([temporal_features, spatial_features], dim=-1)  # (B, N, temporal_dim + spatial_dim)
        combined_reshaped = combined_features.reshape(-1, combined_features.shape[-1])
        
        # 通过KAN进行特征融合
        fused_reshaped = self.kan_fusion(combined_reshaped)  # (B*N, fusion_dim)
        fused_features = fused_reshaped.reshape(B, N, -1)  # (B, N, fusion_dim)
        
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features


class LLaMA3_Lora_block_with_KAN(nn.Module):
    """
    Modified LLaMA3 block with KAN-based feature fusion
    """
    def __init__(self, DEVICE, num_for_predict, d_model, temporal_dim, spatial_dim):
        super(LLaMA3_Lora_block_with_KAN, self).__init__()
        
        # KAN特征融合模块
        self.kan_fusion = KANFeatureFusion(DEVICE, temporal_dim, spatial_dim, d_model)
        
        self.pretrained_model_path = '/llama3/model/'      # LLaMA3模型加载
        config = LlamaConfig.from_pretrained(self.pretrained_model_path)
        self.llama3 = LlamaModel.from_pretrained(
            self.pretrained_model_path, 
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA配置
        self.peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        
        # 准备模型并进行LoRA适配
        self.llama3 = prepare_model_for_kbit_training(self.llama3)
        self.lora_llama3 = get_peft_model(self.llama3, self.peft_config)
        
        # 输出层
        self.out_layer = nn.Linear(config.hidden_size, num_for_predict)
        for layer in (self.lora_llama3, self.out_layer):
            layer.to(DEVICE)

    def forward(self, temporal_features, spatial_features):
        """
        Forward pass with KAN-based feature fusion
        
        Args:
            temporal_features: (B, N, temporal_dim) - temporal features
            spatial_features: (B, N, spatial_dim) - spatial features
            
        Returns:
            output: (B, N, T) - prediction output
        """
        # 使用KAN融合时空特征
        fused_features = self.kan_fusion(temporal_features, spatial_features)  # (B, N, d_model)
   
        outputs = self.lora_llama3(inputs_embeds=fused_features, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (B, N, hidden_size)
        output = self.out_layer(hidden_states)  # (B, N, T)

        return output


class STATPLLM_with_KAN(nn.Module):
    """
    Enhanced STATPLLM model with KAN-based spatio-temporal feature fusion
    """
    def __init__(self, DEVICE, num_of_vertices, num_of_timesteps, adj, in_channels, nb_filter, time_strides, d_model, num_for_predict):
        super(STATPLLM_with_KAN, self).__init__()
        
        # STA-TPLLM的时空分步提取
        self.TE = TimeEmb(DEVICE, num_of_vertices, num_of_timesteps)
        self.TE_conv = nn.Conv2d(in_channels, nb_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.SA = Spatial_Attention(DEVICE, num_of_vertices, num_of_timesteps)
        self.SE = Graph_Conv_with_SAtt(DEVICE, num_of_vertices, num_of_timesteps, adj, in_channels=in_channels, out_channels=nb_filter)

        self.Wv2 = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))
        self.b1 = nn.Parameter(torch.FloatTensor(1, nb_filter, num_of_vertices, num_of_timesteps).to(DEVICE))
        self.b2 = nn.Parameter(torch.FloatTensor(1, nb_filter, num_of_vertices, num_of_timesteps).to(DEVICE))
        self.residual_conv = nn.Conv2d(in_channels, nb_filter, kernel_size=(1, 1), stride=(1, time_strides))
        
        self.GELU = nn.GELU()
        self.leaky_relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(nb_filter)
        
        # 输入卷积和KAN-LLaMA模块
        self.input_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, 1))
        
        # 修改为包含KAN的LLaMA模块
        temporal_feature_dim = nb_filter  # 时间特征维度
        spatial_feature_dim = nb_filter   # 空间特征维度
        self.llama3_with_kan = LLaMA3_Lora_block_with_KAN(
            DEVICE, num_for_predict, d_model, temporal_feature_dim, spatial_feature_dim
        )
        
        self.reset_parameters()

        for layer in (self.TE, self.TE_conv, self.SA, self.SE, self.residual_conv, 
                     self.GELU, self.leaky_relu, self.ln, self.input_conv):
            layer.to(DEVICE)

    def reset_parameters(self):
        init.kaiming_uniform_(self.Wv2)
        init.kaiming_uniform_(self.b1)
        init.kaiming_uniform_(self.b2)

    def extract_temporal_features(self, x):
        """提取时间特征"""
        # Time Embedding
        x_TE = self.TE(x)  
        x_TE = F.relu(self.TE_conv(x_TE.permute(0, 2, 1, 3)) + self.b1)  # B 64 N T
        
        return x_TE

    def extract_spatial_features(self, x):
        """提取空间特征"""
        # Graph Embedding
        SA = self.SA(x)  
        
        x_s = torch.matmul(self.Wv2, x.permute(0, 2, 1, 3)) 
        x_s = x_s.permute(0, 2, 1, 3)  # BN1T
        
        x_SE = self.SE(x_s, SA) 
        x_SE = F.relu(x_SE.permute(0, 2, 1, 3) + self.b2)
        
        return x_SE

    def forward(self, x):
        '''
        :param x: B, N, in_channels, T
        :return: B, N, T
        '''
        # 提取时空特征
        x_TE = self.extract_temporal_features(x) 
        x_SE = self.extract_spatial_features(x) 
        
        # 时空特征输入
        temporal_input = x_TE.permute(0, 2, 1, 3)  # B N 64 T
        temporal_input = temporal_input.mean(dim=-1)  # B N 64 
        spatial_input = x_SE.permute(0, 2, 1, 3)  # B N 64 T
        spatial_input = spatial_input.mean(dim=-1)  # B N 64 
        
        # 使用KAN融合时空特征并输入LLaMA3
        output = F.relu(self.llama3_with_kan(temporal_input, spatial_input)) 

        return output


def make_STATPLLM_with_KAN_model(DEVICE, len_input, time_strides, in_channels, nb_chev_filter, adj_mx, num_of_vertices, d_model, num_for_predict):
    """
    Factory function to create STATPLLM model with KAN fusion
    """
    model = STATPLLM_with_KAN(
        DEVICE=DEVICE,
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

