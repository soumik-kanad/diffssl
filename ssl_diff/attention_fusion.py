import torch
import torch.nn as nn
from .attention_head import AttentionHead, LambdaLayer
from ssl_diff import Head
from einops import rearrange 

class AttentionFusion(nn.Module):
    def __init__(self, args, feature_dim_dict, fature_size_dict=None):
        super(AttentionFusion, self).__init__()
        attention_dims = int(args.fusion_arc.split(',')[0].strip().split(':')[2])
        pre_layer = {}
        for b in set(args.first_fw_b_list + args.second_fw_b_list):
            feat_size = min(fature_size_dict[b], args.pre_pool_size)
            norm = nn.BatchNorm2d(feature_dim_dict[b]) if args.norm_type == "batch" else nn.LayerNorm([feature_dim_dict[b], feat_size, feat_size])
            pre_layer[str(b)] = nn.Sequential(
                nn.AdaptiveAvgPool2d(feat_size),
                norm,
                nn.Conv2d(feature_dim_dict[b], attention_dims, 1),
                LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')),
            )
        self.pre_layer = nn.ModuleDict(pre_layer) 

        self.intra_inter_block_attention = AttentionHead(args.fusion_arc.split("/")[0])
        self.feature_dims = attention_dims * len(args.t_list)
        self.head = nn.Linear(self.feature_dims, args.num_classes)

    def forward(self, first_fw_b_list, first_fw_feat, second_fw_b_list, second_fw_feat, t_list):
        if t_list is None: t_list = [0]  # for other than Diffusion Model
        inter_noise_step_feat = []
        for t_idx, t in enumerate(t_list):
            block_feat = []
            for b_idx, b in enumerate(first_fw_b_list):
                x = self.pre_layer[str(b)](first_fw_feat[t_idx][b_idx])
                block_feat.append(x)
            for b_idx, b in enumerate(second_fw_b_list):
                x = self.pre_layer[str(b)](second_fw_feat[t_idx][b_idx])
                block_feat.append(x)
            x = torch.concat(block_feat, dim=1)
            # print("DEBUG: intra_inter_block_feat.in.shape", x.shape)
            x = self.intra_inter_block_attention(x)
            # print("DEBUG: intra_inter_block_feat.out.shape", x.shape)
            inter_noise_step_feat.append(x)
        x = torch.concat(inter_noise_step_feat, dim=1)
        # print("DEBUG: inter_noise_feat.shape", x.shape)
        x = self.head(x)
        return x


