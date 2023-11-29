import torch
from torch import nn
from einops import rearrange 
import math
from torchinfo import summary 

class AttentionHead(nn.Module):
    """ 
    Class to create an attention head for a neural network. 
    The layer string is a comma separated string of layer definitions.
    The layer definitions are of the form:
    
    Use_CLS_Token:bool:dim
    Pool:k:s
    Reshape 
    Insert_CLS_Token
    Block:dim:num_heads:mlp_ratio:num_blocks
    Extract_CLS_Token
    Dense:in_dim:out_dim:act 

    """
    
    def __init__(self, layer_string): 
        super(AttentionHead, self).__init__()
        self.layer_string = layer_string
        self.layers = self.build_model()

        # Build CLS token based on layer string 
        self.use_cls_token, dim = self.parse_use_cls_token(layer_string.split(',')[0].strip())
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def build_model(self):
        layers = []
        layer_definitions = self.layer_string.split(',')

        for layer_definition in layer_definitions:
            layer_definition = layer_definition.strip()
            if layer_definition.startswith('Attention'):
                dim, num_heads, mlp_ratio, num_blocks = self.parse_block_layer(layer_definition)
                for i in range(num_blocks):
                    layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))
            elif layer_definition.startswith('Flatten'):
                layers.append(nn.Flatten())
            elif layer_definition.startswith('Dense'):
                in_dim, out_dim, activation = self.parse_dense_layer(layer_definition)
                layers.append(nn.Linear(in_dim, out_dim))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'identity':
                    pass
            elif layer_definition.startswith('Pool'):
                pool_size, stride = self.parse_pool_layer(layer_definition)
                layers.append(nn.AvgPool2d(pool_size, stride=stride))
            elif layer_definition.startswith('FullyPool'):
                layers.append(LambdaLayer(lambda x: x.mean(dim=1)))
            elif layer_definition.startswith('Use_CLS_Token'): 
                pass 
            elif layer_definition.startswith('Reshape'):
                layers.append(LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')))
            elif layer_definition.startswith('Insert_CLS_Token'):
                layers.append(LambdaLayer(lambda x: torch.cat((self.cls_token.to(x.device).expand(x.shape[0], -1, -1),
                                                               x), dim=1)))
            elif layer_definition.startswith('Extract_CLS_Token'):
                layers.append(LambdaLayer(lambda x: x[:, 0]))

        return nn.Sequential(*layers)

    def parse_use_cls_token(self, layer_definition):
        layer_definition = layer_definition.split(':')
        use_cls_token = bool(layer_definition[1])
        dim = int(layer_definition[2])
        return use_cls_token, dim
    
    def parse_pool_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')
        pool_size = int(layer_definition[1])
        stride = int(layer_definition[2])
        return pool_size, stride
    
    def parse_block_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')
        dim = int(layer_definition[1])
        num_heads = int(layer_definition[2])
        mlp_ratio = float(layer_definition[3])
        num_blocks = int(layer_definition[4])
        return dim, num_heads, mlp_ratio, num_blocks
    
    def parse_dense_layer(self, layer_definition):
        layer_definition = layer_definition.split(':')
        in_dim = int(layer_definition[1])
        out_dim = int(layer_definition[2])
        activation = layer_definition[3]
        return in_dim, out_dim, activation


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() # I just default set this to identity function
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__": 

    print("NO CLS TOKEN")
    no_cls_layer_strings = [] 
    for num_blocks in [1, 2, 3, 4, 5]:
        layer_string = ""
        layer_string += "Pool:2:2,"
        layer_string += "Reshape,"
        layer_string += f"Attention:1024:8:4:{num_blocks}," # dim, num_heads, mlp_ratio, num_blocks 
        layer_string += "FullPool,"
        layer_string += "Dense:1024:1000:identity"  

        # build model 
        model = AttentionHead(layer_string).cuda()
        # print(model)

        if num_blocks == 1:
            x = torch.randn(1, 1024, 16, 16).cuda()
            out = model(x)
            print(out.shape)

        # number of parameters
        print("Num Blocks: ", num_blocks, ", Num Params: ", sum(p.numel() for p in model.parameters() 
                                                                if p.requires_grad))
        no_cls_layer_strings.append(layer_string)

    print("CLS TOKEN")
    cls_layer_strings = [] 
    for num_blocks in [1, 2, 3, 4, 5]:
        layer_string = ""
        layer_string += "Use_CLS_Token:True:1024,"
        layer_string += "Pool:2:2,"
        layer_string += "Reshape,"
        layer_string += "Insert_CLS_Token,"
        layer_string += f"Attention:1024:8:4:{num_blocks}," # dim, num_heads, mlp_ratio, num_blocks 
        layer_string += "Extract_CLS_Token,"
        layer_string += "Dense:1024:1000:identity"  

        # build model 
        model = AttentionHead(layer_string).cuda()
        # print(model)

        if num_blocks == 1:
            x = torch.randn(1, 1024, 16, 16).cuda()
            out = model(x)
            print(out.shape)

        # number of parameters
        print("Num Blocks: ", num_blocks, ", Num Params: ", sum(p.numel() for p in model.parameters() 
                                                                if p.requires_grad))
        cls_layer_strings.append(layer_string)

    print(no_cls_layer_strings)
    print(cls_layer_strings)