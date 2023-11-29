from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
import  torch 
from torch import nn
import torch.nn.functional as F
from .const import DM_FEAT_DIM_DICT
from guided_diffusion.nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 1, padding=0),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class DiffSSLModelFeedbackFusion(nn.Module):

    """ SSL model with feedback loop between the decoder features of the UNet and the 
        encoder features of the UNet
    """
    def __init__(self, encoder, diffusion, head, device, mode='freeze', 
                 feedback_arch="C_B_R_C", use_feedback=False, feedback_b_list=None, first_fw_b_list=None, second_fw_b_list=None):
        
        super().__init__()
        self.encoder = encoder
        self.diffusion = diffusion
        self.head = head
        self.use_feedback = use_feedback
        self.feedback_b_list = feedback_b_list
        self.first_fw_b_list = first_fw_b_list
        self.second_fw_b_list = second_fw_b_list
        self.mode = mode
        assert self.mode in ['freeze', 'update', 'mult_fpn', 'add_fpn', 'multi_scale_freeze', "finetune"], f"Mode {self.mode} not supported"
        
        if self.mode == 'freeze' and not use_feedback: 
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # including fusion finetune, feedback finetune, feedback freeze
            # print("=======Freezed param=======")
            frozen_decoder_idx = max(self.first_fw_b_list + self.second_fw_b_list) - 19 # -19 to convert block idx to decoder idx
            for name, param in self.encoder.named_parameters():
                if name.startswith("out."):
                    param.requires_grad = False
                    # print(name)
                elif name.startswith("output_blocks"):
                    if int(name.split(".")[1]) >= frozen_decoder_idx:
                        param.requires_grad = False
                        # print(name)

        self.device = device        
       
        if use_feedback:
            """ 
            generate feedback layers
            Feedback Architecture: feedback_arch = "C_B_R_C" = Conv, BN, ReLU, Conv
            """
            feedback_layers = []
            for feedback_b in self.feedback_b_list:
                in_dim = DM_FEAT_DIM_DICT[feedback_b]
                out_dim = DM_FEAT_DIM_DICT[38-feedback_b]
                sequential_model_lst = self.make_layers(feedback_arch, in_dim, out_dim)
                feedback_layers.append(nn.Sequential(*sequential_model_lst))
            self.feedback_layers = nn.ModuleList(feedback_layers)
    
    def make_layers(self, feedback_arch, in_dim, out_dim):
        sequential_model_lst = [] 
        for j in range(len(feedback_arch)):
            if feedback_arch[j] == "Res":
                """ Use first block to change in_dim to out_dim and then the rest operate on out_dim """
                if j == 0: # if the first resblock
                    sequential_model_lst.append(ResBlock(in_dim, dropout=0.0, out_channels=out_dim, use_conv=False))
                else: # if the last resblock
                    sequential_model_lst.append(ResBlock(out_dim, dropout=0.0, out_channels=out_dim, use_conv=False))
            elif feedback_arch[j] == "R":
                sequential_model_lst.append(nn.ReLU(inplace=True))
            elif feedback_arch[j] == "B":
                sequential_model_lst.append(nn.BatchNorm2d(out_dim))
            elif feedback_arch[j] == "C":
                """ Use first conv to change in_dim to out_dim and then the rest operate on out_dim """
                if j == 0:
                    sequential_model_lst.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False))
                else:
                    sequential_model_lst.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=False))
            elif feedback_arch[j] == "C2":
                """ Operate on in_dim the entire time and then for the last conv, change in_dim to out_dim """
                if j == len(feedback_arch) - 1:
                    sequential_model_lst.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False))
                else:
                    sequential_model_lst.append(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False))
            elif feedback_arch[j] == "S": 
                sequential_model_lst.append(nn.SiLU(inplace=True))
            elif feedback_arch[j] == "G":
                # want to use this as a group norm layer
                sequential_model_lst.append(nn.GroupNorm(num_groups=32, num_channels=out_dim, dtype=self.encoder.dtype))
        return sequential_model_lst

    
    def generate_feedback(self, features):
        """ generate feedback features from decoder features """
        feedback_features = []
        for idx, b in enumerate(self.feedback_b_list):
            feedback_features.append(self.feedback_layers[idx](features[b-1]))
        return feedback_features

    def forward(self, x, t, unet_model_kwargs={}):
        first_fw_feat = []
        second_fw_feat = []
        for t_ in t:
            t_ = t_*torch.ones(x.shape[0],).long().to(self.device)
            x_start = x.to(self.device)
            x_start = x_start.type(torch.float16) if self.use_fp16 else x_start.type(torch.float32)
            noise = torch.randn_like(x_start)

            x_t = self.diffusion.q_sample(x_start, t_, noise=noise)

            # encoder_features = self.encoder.get_encoder_features(x_t, self.diffusion._scale_timesteps(t), **unet_model_kwargs)
            # print([x.shape for x in encoder_features])

            """ extract encoder features and decoder features depending on the mode """
                    
            if self.use_feedback:
                with torch.no_grad():
                    # TODO : getting all features wastes GPU memory
                    encoder_features, _, mid_feature, decoder_features = self.encoder.get_all_features(x_t, 
                                                self.diffusion._scale_timesteps(t_), 0,
                                                ['encoder_features', 'resume_encoder_feature', 'mid_feature', 'decoder_features'], 
                                                **unet_model_kwargs)
                features = encoder_features + mid_feature + decoder_features
                first_fw_feat.append([features[b-1].detach().float() for b in self.first_fw_b_list])
            else:
                block_feat_lst = self.encoder.get_multiple_features(x_t, 
                                        self.diffusion._scale_timesteps(t_),
                                        block_num_lst = self.first_fw_b_list, 
                                        **unet_model_kwargs)
                first_fw_feat.append([block_feat.float() for block_feat in block_feat_lst])
                
                    

                    
            if self.use_feedback: # use feedback 
                """ generate feedback features from decoder features """
                feedback_features = self.generate_feedback(features)
                feedback_features = feedback_features[::-1] # reverse the list of feedback features 

                """ generate the final features based on the mode """
                block_feat_list = self.encoder.get_multiple_features_with_specified_feedback(x=x_t, 
                    timesteps=self.diffusion._scale_timesteps(t_), 
                    block_num_lst=self.second_fw_b_list, 
                    feedback_features=feedback_features, # list of features [0: len(input_blocks) - feedback_starting_point]
                    feedback_b_list=self.feedback_b_list,
                    **unet_model_kwargs)
                second_fw_feat.append([block_feat.float() for block_feat in block_feat_list])

        x = self.head(self.first_fw_b_list, first_fw_feat, self.second_fw_b_list, second_fw_feat, t)
        
        return x
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.encoder.convert_to_fp16()
        if self.use_feedback:
            for idx in range(len(self.feedback_b_list)):
                self.feedback_layers[idx].apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False 
        self.encoder.convert_to_fp32()
        if self.use_feedback:
            for idx in range(len(self.feedback_b_list)):
                self.feedback_layers[idx].apply(convert_module_to_f32)
