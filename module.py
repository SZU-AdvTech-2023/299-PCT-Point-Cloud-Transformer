import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from util import index_points, knn_point, sample_and_knn_group,sample_and_ball_group

class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]
        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))

        # residual
        x = x + x_s

        return x

class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_ball_group(s=self.s,radius=0.5, n=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_xyz, new_feature


class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512, 256]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.add = SA(122)
        self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg2 = SG(s=samples[1], in_channels=256, out_channels=256)
    
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]++
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]adadas

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
        coords, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]

        return coords,features2

class low_attention(nn.Module):  
    def __init__(self, channels):
        super(low_attention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self,x):
        """
        Input  x: [B, D, low_N]        
        Output x: [B, D, low_N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)#change shape
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # [B, low_n, low_n]

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))                                                                                                                                                                                                                                                                                           
        x = x + x_r
        return x,attention

class full_attention(nn.Module):
  
    def __init__(self, channels):
        super(full_attention, self).__init__()
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  

        # fps_idx:[B,low_n]
        # knn_idx:[B,low_n,k]
        # low_attention:[B,low_n,low_n]
        # x:[B,D,n]

        
    def forward(self, x,low_attention,fps_idx,knn_idx): 
        x_v = self.v_conv(x)                
        topk_values, topk_indices = torch.topk(low_attention, k=16, dim=-1)
        #topk_values:[B,low_n,topk] #topk_indices:[B,low_n,topk]
        #top_k=KNN
        attention1=torch.zeros((x.shape[0],(x.shape[2]),(x.shape[2]))).to('cuda')#[B,N,top k]
        B, _ = x.shape[0], x.shape[2]
        low_n, top_k = topk_values.shape[1], topk_values.shape[2]
        # 将 fps_idx 扩展为与 topk_indices 相同的形状
        expanded_fps_idx = fps_idx.unsqueeze(-1).expand(-1, -1, topk_indices.shape[2])
        # 现在，fps_idx 和 topk_indices 具有相同的维度数
        # 使用 expanded_fps_idx 和 topk_indices 来转换全局索引
        global_topk_indices = torch.gather(expanded_fps_idx, 1, topk_indices)
        # 批量更新 attention
        # 使用 global_topk_indices 作为列索引，expanded_fps_idx 作为行索引        
        attention1[torch.arange(B)[:, None, None], expanded_fps_idx, global_topk_indices] = topk_values
        # 获取 KNN 索引
        # 首先为每个 topk_indices 找到对应的 KNN 索引
        knn_indices = torch.gather(knn_idx, 1, topk_indices)
        # 扩展 topk_values 以匹配 KNN 索引
        expanded_topk_values = topk_values.unsqueeze(-1).expand(-1, -1, -1, knn_idx.size(-1))
        # 扩展 batch 索引以匹配其他维度
        batch_indices = torch.arange(B, device=attention1.device)[:, None, None, None].expand(-1, low_n, top_k, knn_idx.size(-1))
        # 扩展 fps 索引以匹配 KNN 的数量
        expanded_fps_idx = expanded_fps_idx.unsqueeze(-1).expand(-1, -1, -1, knn_idx.size(-1))
        # 扩展 topk_values 以匹配 KNN 索引
        expanded_topk_values = expanded_topk_values.expand(-1, -1, -1, knn_idx.size(-1))

        knn_indices = knn_indices.unsqueeze(-1).expand(-1, -1, -1, knn_idx.size(-1))
        # 批量更新 attention              
        attention1[batch_indices, expanded_fps_idx, knn_indices] = expanded_topk_values
        
        x_r = torch.bmm(x_v, attention1)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))                                                                                                                                                                                                                                                                                           
        x = x + x_r
        return x
    
'''
        for b in range(x.size(0)):
            for i in range(topk_values.size(1)):
                full_idx=fps_idx[b][i]
                for k in range(topk_values.size(2)):                  
                    target_idx=fps_idx[b][topk_indices[b][i][k]]
                    atten=topk_values[b][i][k]
                    attention[b][full_idx][target_idx]=atten
                    for knn in range(knn_idx.size(2)):
                        knn_index=knn_idx[b][topk_indices[b][i][k]][knn]                       
                        attention[b][full_idx][knn_index]=atten
'''
if __name__ == '__main__':
   
    x = torch.rand(10, 3, 1024).to('cuda')
    neighbor_embedding=NeighborEmbedding(samples=[512,256]).to('cuda')
    coords,x = neighbor_embedding(x)#embedding
    batch_size = coords.shape[0]
    coords = coords.contiguous()
    # FPS sampling
    low_n=64
    fps_idx = pointnet2_utils.furthest_point_sample(coords, low_n).long()  # [B, low_n]   
    low_coords = index_points(coords, fps_idx)           # [B, low_n, 3]
    low_x = index_points(x.permute(0, 2, 1), fps_idx)                     # [B, low_n, D]
    low_x=low_x.permute(0, 2, 1)
    #print(fps_idx.size())

    # K-nn grouping
    k=16
    knn_idx = knn_point(k, coords,low_coords)            # [B, low_n, k]
    low_oa1= low_attention(256).to('cuda') 
    high_oa1= full_attention(256).to('cuda')    
    low_x,atten = low_oa1(low_x)
    x=high_oa1(x,atten,fps_idx,knn_idx) 



