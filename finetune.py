import argparse
import torch
import numpy as np
import sys
import os 
import glob

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb 
from tqdm import tqdm 

from ssl_diff import DiffSSLModelFeedbackFusion
from ssl_diff import AttentionFusion, Head
from ssl_diff import DM_FEAT_DIM_DICT,DM_FEAT_SIZE_DICT

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util #, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def train(model, lr, e, bs, train_dataloader, test_dataloader, args, checkpoint_dict=None):


    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # apply optimizer only on the head and feedback layers
    use_feedback = len(args.second_fw_b_list) > 0
    if args.distributed:
        optimized_params_lst = [{'params': model.module.head.parameters()}]
        if use_feedback:
            optimized_params_lst.append({'params': model.module.feedback_layers.parameters()})
            if args.mode == 'update':
                optimized_params_lst.append({'params': model.module.update_blocks.parameters()})
            if args.mode == 'add_fpn' or args.mode == 'mult_fpn':
                optimized_params_lst.append({'params': model.module.fpn_blocks.parameters()})
        if args.mode == "finetune":
            optimized_params_lst.append({'params': model.module.encoder.parameters()})
    else: 
        optimized_params_lst = [{'params': model.head.parameters()}]
        if use_feedback:
            optimized_params_lst.append({'params': model.feedback_layers.parameters()})
            if args.mode == 'update':
                optimized_params_lst.append({'params': model.update_blocks.parameters()})
            if args.mode == 'add_fpn' or args.mode == 'mult_fpn':
                optimized_params_lst.append({'params': model.fpn_blocks.parameters()})
        if args.mode == "finetune":
            optimized_params_lst.append({'params': model.encoder.parameters()})

    optimizer = torch.optim.SGD(optimized_params_lst, lr=lr)

    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, 0.1)

    if checkpoint_dict is not None:
        print(f"Loading model, optimizer, and scheduler from checkpoint from!")

        model.module.head.load_state_dict(checkpoint_dict['model_head'])
        if use_feedback:
            print("Loading feedback layers")
            model.module.feedback_layers.load_state_dict(checkpoint_dict['model_feedback'])
        if args.mode == 'update':
            print("Loading update blocks")
            model.module.update_blocks.load_state_dict(checkpoint_dict['model_update'])
        elif args.mode == 'add_fpn' or args.mode == 'mult_fpn':
            print("Loading fpn blocks")
            model.module.fpn_blocks.load_state_dict(checkpoint_dict['model_fpn'])

        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        start_epoch = checkpoint_dict['epoch']
    else:
        start_epoch = 0
        
    losses = []
    model.train()
    batch_num = 0

    for i in range(start_epoch, e):
        for batch in (tqdm(train_dataloader, total=len(train_dataloader))):

            # # measure execution time in pytorch 
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            
            # start.record() 
           
            imgs, extra = batch #next(train_dataloader)
            imgs = imgs.to(dist_util.dev())
            targets = extra["y"].to(dist_util.dev())
           
            # end.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print("Inputs: ", start.elapsed_time(end))

            # start.record()

            output = model(imgs, args.t_list)

            # end.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print("Forward: ", start.elapsed_time(end))


            # start.record()  
            #calculate loss
            loss = loss_fn(output, targets)
            # end.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print("Loss: ", start.elapsed_time(end))

            #backprop
            # start.record()
            optimizer.zero_grad()
            loss.backward()

            # store 'module.encoder.time_embed.0.bias' weight
            # import pdb;x pdb.set_trace()
            # print(old - model.module.encoder.time_embed[0].bias.clone().detach())
            # old = model.module.encoder.time_embed[0].bias.clone().detach()

            optimizer.step()
            # end.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print("Backward: ", start.elapsed_time(end))

            # start.record()
            if len(losses) == 100:
                losses = losses[1:]
            losses.append(loss.item())

            if dist_util.is_main_process():
                if (batch_num + 1) % 100 == 0:
                    print(f'Epoch: {i+1}/{e}, Batch Num: {batch_num+1}: Loss: {np.mean(losses):0.6f}', flush=True)
                    if args.use_wandb:
                        wandb.log({"Loss/train": np.mean(losses), "epoch": (batch_num+1) / len(train_dataloader)})
            batch_num += 1
            # end.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print("Logging: ", start.elapsed_time(end))

        scheduler.step()
        if (i + 1) % args.eval_interval == 0:
            test(model, test_dataloader, args, 'Val (Test)', i+1)

        # Save checkpoint every epoch
        if dist_util.is_main_process():
            save_file = os.path.join(args.output_dir, f'epoch_latest.pth')
            print(f"Saving checkpoint @ Epoch: {i+1} to {save_file}")
            save_dict ={
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': i+1
                        }
            save_dict['model_head'] = model.module.head.state_dict()
            if use_feedback:
                save_dict['model_feedback'] = model.module.feedback_layers.state_dict()
            if args.mode == 'update':
                save_dict['model_update'] = model.module.update_blocks.state_dict()
            elif args.mode == 'add_fpn' or args.mode == 'mult_fpn':
                save_dict['model_fpn'] = model.module.fpn_blocks.state_dict()
            

            # torch.save(save_dict, save_file)
            torch.save(save_dict, os.path.join(args.output_dir, f'latest.pth'))


# https://discuss.pytorch.org/t/ddp-evaluation-gather-output-loss-and-stuff-how-to/130593/2
def sync_tensor_across_gpus(t):
    # t needs to have dim 0 for torch.cat below. 
    # if not, you need to prepare it.
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu. 
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with 
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the 
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html). 
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for 
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in 
   # the doc is  vague... 
    return torch.cat(gather_t_tensor, dim=0)

def test(model, dataloader, args, split='Test', epoch=0):
    model.eval()
    num_correct = 0
    total = 0

    num_val_batches = len(dataloader)

    with torch.no_grad():

        for batch in tqdm(dataloader, total=num_val_batches):
            imgs, extra = batch
            imgs = imgs.to(dist_util.dev())
            targets = extra["y"].to(dist_util.dev())
            output = model(imgs, args.t_list)
            pred = torch.argmax(output, dim=1)
            # print("Pred:", pred)
            # print("Targets:", targets)
            num_correct += (pred == targets).sum()
            total += pred.shape[0]
            # print(dist_util.get_rank(), total, num_correct)
            
            # print("Acc now:", num_correct/total)
        all_num_correct = sync_tensor_across_gpus(torch.tensor(num_correct).to(dist_util.dev()).reshape(1))
        all_total = sync_tensor_across_gpus(torch.tensor(total).to(dist_util.dev()).reshape(1))
    if dist_util.is_main_process():
        num_correct = all_num_correct.sum().item()
        total = all_total.sum().item()
        if args.use_wandb:
            wandb.log({f"Accuracy/{split}": num_correct / total, "epoch": epoch})
        print(f'{split} accuracy: {num_correct / total}, Num correct: {num_correct}, Total: {total}')

def create_argparser():
    defaults = dict(
        # dataset
        data_dir="",
        val_data_dir="",
        num_classes=50,
        
        # training setting
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=0,
        epochs = 50,
        lr=1e-2, # use 1e-2 for freeze, 1e-3 for finetune
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        use_fp16=False,
        fp16_scale_growth=1e-3,
        mode='freeze', # "freeze" or "finetune" for backbone

        # feedback&fusion
        head_type="attention", # "linear" or "attention"
        head_arc='', #can be "h1_h2_h3" to use mlp head(when head_type=="linear")
        norm_type="",  # ["batch", "layer", ""],
        pre_pool_size=16, # pooling size before attention or linear head
        fusion_arc="", # architecture of attention head
        feedback_arch='C_B_R', # architecture for feedback network(Conv-BatchNorm-ReLU)
        
        checkpoint_path='', # encoder path

        # add distributed training args
        dist_url='env://',
        dist_backend='nccl',
        world_size=1,

        # log
        output_dir='./output',  
        resume_checkpoint="",
        log_interval=10,
        save_interval=10000,
        eval_interval=5,
        only_eval=False,
        wandb_run_name=None,
        use_wandb=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0, 
                    help="For distributed training.")
    parser.add_argument("--t_list", nargs='+', required=True, type=int, help="list of noise step t to use")
    parser.add_argument("--first_fw_b_list", nargs='+', required=True, type=int, help="list of feature block to use from first forward(-1 if not used)")
    parser.add_argument("--second_fw_b_list", nargs='+', required=True, type=int, help="list of feature block to use from second forward(=feedback)(-1 if not used)")
    parser.add_argument("--feedback_b_list", nargs='+', required=True, type=int, help="list of feature block for feedback(-1 if not used)")
    add_dict_to_argparser(parser, defaults)
    return parser


def main():

    print('Reading args')
    args = create_argparser().parse_args()
    args.device = 'cuda'

    if args.first_fw_b_list[0] == -1:
        args.first_fw_b_list = [] 
    if args.second_fw_b_list[0] == -1:
        args.second_fw_b_list = [] 
    use_feedback = len(args.second_fw_b_list) > 0
    if args.feedback_b_list[0] == -1:
        args.feedback_b_list = [] 
        assert use_feedback==False, "blocks for feedback are not specified"
    if args.head_type == "linear":
        assert len(args.first_fw_b_list) == 1 and len(args.t_list) == 1 and len(args.second_fw_b_list) == 0, "linear head cannot be used for feedback/fusion"

    print('Setting up dist')
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    # args.distributed = True
    args.device = torch.device(args.device)
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print("")
        print("Init distributed training on local rank {} ({}), world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        torch.distributed.barrier()

    if dist_util.is_main_process(): 
        print(f'args: {args}')
        exist_ok = len(glob.glob(os.path.join(args.output_dir, "*"))) == 0 or (args.output_dir == os.path.dirname(args.resume_checkpoint)) or (os.path.basename(args.output_dir) == "debug")
        os.makedirs(args.output_dir, exist_ok=exist_ok)
        print('Creating model')

    if args.head_type =="linear":
        head = Head(args, DM_FEAT_DIM_DICT,DM_FEAT_SIZE_DICT)
    elif args.head_type == "attention":
        head = AttentionFusion(args, DM_FEAT_DIM_DICT,DM_FEAT_SIZE_DICT)
    else:
        raise NotImplementedError


    encoder, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    #print(encoder)

    if dist_util.is_main_process(): print('Loading model')
    encoder.load_state_dict(
            dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
        )
    
    if dist_util.is_main_process(): print('Adding head')

    model = DiffSSLModelFeedbackFusion(encoder, diffusion, head, 
                    device=dist_util.dev(), 
                    mode = args.mode, 
                    feedback_arch=args.feedback_arch,
                    feedback_b_list=args.feedback_b_list,
                    first_fw_b_list=args.first_fw_b_list,
                    second_fw_b_list=args.second_fw_b_list,
                    use_feedback=use_feedback).to(dist_util.dev())

    
    #the whole model should be converted to fp16
    if args.use_fp16:
        if dist_util.is_main_process(): print("Converting to fp16")
        model.convert_to_fp16()

    # Number of Parameters in Model 
    # params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # print("Number of Parameters: ", params)

    # print number of parameters in model
    if dist_util.is_main_process():
        # print(f"Number of Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        num_param = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        if use_feedback:
            num_param += sum(p.numel() for p in model.feedback_layers.parameters() if p.requires_grad)
        if args.mode == "finetune":
            num_param += sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

        print("\n====================\n")
        print("Number of Parameters:")
        for name, p in model.head.named_parameters():
            if p.requires_grad:
                print(f"{name}: {p.numel()}({p.numel()/num_param*100:.2f}%)")
        if use_feedback:
            for name, p in model.feedback_layers.named_parameters():
                if p.requires_grad:
                    print(f"{name}: {p.numel()}({p.numel()/num_param*100:.2f}%)")
        if args.mode == "finetune":
            for name, p in model.encoder.named_parameters():
                if p.requires_grad:
                    print(f"{name}: {p.numel()}({p.numel()/num_param*100:.2f}%)")
        print("Total Number of Parameters:", num_param)
        # print("Input Dim to Head:", fusion.feature_dims)
        print("\n====================\n")

    # Needed for creating correct EMAs and fp16 parameters.
    if args.distributed:
        dist_util.sync_params(model.parameters())

        # DDP 
        model = DDP(model, device_ids=[dist_util.dev()], 
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
        )
    
    # wandb
    if dist_util.is_main_process():
        if args.use_wandb:
            args.learning_params = num_param
            wandb.init(project="diffusion_ssl-4", config=args, name=args.wandb_run_name)
            wandb.watch(model, log='gradients', log_freq=100)
        
    # Load checkpoint
    # args.resume_checkpoint = os.path.join(args.output_dir, 'latest.pth')
    if os.path.exists(args.resume_checkpoint):
        if dist_util.is_main_process(): print("Loading checkpoint from ", args.resume_checkpoint)
        checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
    else: 
        checkpoint = None
    
    if not args.only_eval:
        data = load_data(
            data_dir=args.data_dir, #args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            deterministic=False,
            random_crop=True,
            random_flip=True
        )

        val_data = load_data(
            data_dir=args.val_data_dir, #args.val_data_dir, 
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            deterministic=True,
            random_crop=False,
            random_flip=False
        )

    test_data = load_data(
        data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        deterministic=True,
        random_crop=False,
        random_flip=False
    )

    if not args.only_eval:
        train(model, args.lr, args.epochs, args.batch_size, data, test_data, args, checkpoint)
        test(model, val_data, args, 'Train')
    else: 
        if checkpoint is not None:
            print(f"Loading model, optimizer, and scheduler from checkpoint from!")
            model.module.head.load_state_dict(checkpoint['model_head'])
            model.module.feedback_layers.load_state_dict(checkpoint['model_feedback'])
        test(model, test_data, args, 'Test')

if __name__ == "__main__":
    main()
