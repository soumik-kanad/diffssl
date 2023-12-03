# Diffusion SSL (diffssl)
### Code for the paper "[Do text-free diffusion models learn discriminative visual representations?](https://arxiv.org/abs/2311.17921)". <br />
(This work supersedes "[Diffusion Models Beat GANs on Image Classification](https://arxiv.org/abs/2307.08702)".)

Self-Supervised Learning using Unconditional Diffusion Models.

## Setting up the environment
1. Clone this repository and navigate to it in your terminal. 
1. install python==3.9
1. `pip install -e .`  
This should install the `guided_diffusion` python package that the scripts depend on.
1. `pip install -r requirements.txt` 

## Running the scripts
- Download [this](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) checkpoint from the original guided-diffusion repo and place it in 
`checkpoints/256x256_diffusion_uncond.pt`.

- The scripts for different modes are located in the `scripts` directory. The scripts can be run using the following commands. Make sure to specify `TRAIN_DATA` and `VAL_DATA` in the scripts before running. 
    - GD(L): `bash scripts/linear.sh`
    - Attention head: `bash scripts/attention.sh`
    - DifFormer: `bash scripts/fusion.sh`
    - DifFeed: `bash scripts/feedback.sh`

For more information on the arguments, refer to the `finetune.py:create_argparser()`.
