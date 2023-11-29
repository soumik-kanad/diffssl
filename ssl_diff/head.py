from torch import nn
import torch

class Head(nn.Module):
    def __init__(self, args, feature_dim_dict, feature_size_dict):
        super().__init__()
        self.fcs = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d(args.pre_pool_size)
        feature_dims = feature_dim_dict[args.first_fw_b_list[0]] * args.pre_pool_size * args.pre_pool_size
        if args.head_arc == '':
            self.fcs.append(nn.Linear(feature_dims, args.num_classes))
        else:
            if '_' in args.head_arc:
                hidden_dims = args.head_arc.split('_')
                self.fcs.append(nn.Linear(feature_dims, int(hidden_dims[0])))
                last_hidden = int(hidden_dims[0])
                for hidden_dim in hidden_dims[1:]:
                    self.fcs.append(nn.Linear(last_hidden, int(hidden_dim)))
                    last_hidden = int(hidden_dim)
                self.fcs.append(nn.Linear(last_hidden, args.num_classes))
            else:
                self.fcs.append(nn.Linear(feature_dims, int(args.head_arc)))
                self.fcs.append(nn.Linear(int(args.head_arc), args.num_classes))
                
    def forward(self, first_fw_b_list, first_fw_feat, second_fw_b_list, second_fw_feat, t_list):
        x = first_fw_feat[0][0]
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        if len(self.fcs) == 1:
            return self.fcs[0](x)
        else:
            for fc in self.fcs[:-1]:
                x = nn.functional.relu(fc(x))
            return self.fcs[-1](x)
