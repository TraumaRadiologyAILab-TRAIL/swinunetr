import argparse
import os
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from pathlib import Path
from utils import *
from transformers import get_linear_schedule_with_warmup
import random
from dataset import *
from loss import *
from train_helper import *
from swin import *
from dali_dataset import *
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
def set_seed(seed, rank):
    random.seed(seed + rank)  # Different seed per GPU
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cudnn.deterministic = True
def main():
    parser = argparse.ArgumentParser('SwinUNETR Training', add_help=False)
    # Model parameters
    parser.add_argument('--model_name', default='swinunetr', type=str)
    parser.add_argument('--img_size', nargs='+', default=[512, 512, 512], type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=3, type=int)
    parser.add_argument('--feature_size', default=48, type=int)
    parser.add_argument('--use_checkpoint', action='store_true')

    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--print_freq', default=10, type=int)

    # Optimizer parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_epochs', default=20, type=int)
    parser.add_argument('--warmup_steps', default=-1, type=int)

    # Loss parameters
    parser.add_argument('--dice_weight', default=1.0, type=float)
    parser.add_argument('--focal_weight', default=1.0, type=float)
    parser.add_argument('--contrastive_weight', default=0.1, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # DDP parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Other parameters
    parser.add_argument('--nGPU', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    
    args = parser.parse_args()


    #setting up variables from parser and setting up seed
    gpu_id = int(os.environ['LOCAL_RANK'])
    setup(gpu_id, args.nGPU)
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.epochs
    num_gpu = args.nGPU


    import pickle
    with open('/home/ankush/detector/first_stage/image_dict_updated_cases.pickle', 'rb') as f:
        image_dict = pickle.load(f) 
    with open('/home/ankush/detector/first_stage/series_dict_updated_cases.pickle', 'rb') as f:
        series_dict = pickle.load(f) 
    # train_df, val_df, test_df = split_data(training_folds=['0', '1', '2','3'], val_folds=['4'], 
    #                                                                   test_fold=['temporal validation'], image_dict=image_dict, series_dict=series_dict)
    # print(f"Train Length: {len(train_df)}, Val Length: {len(val_df)}")
    #next time just load
    train_df = pd.read_csv('train_df.csv')
    #fileter the train df with only row['htx] == 1 or row['mh'] == 1
    train_df = train_df[(train_df['htx'] == 1) | (train_df['mh'] == 1)]
    val_df = pd.read_csv('val_df.csv')
    test_df = pd.read_csv('test_df.csv')
    print(f"Train Length: {len(train_df)}, Val Length: {len(val_df)}")
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    args.device = device
    seed = 42 + gpu_id
    set_seed(seed, gpu_id)

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # # Create DataLoaders
    # train_dataset = MultiTaskMedicalDataset(train_df, None)
    # train_sampler = DistributedSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     collate_fn = custom_collate_stratified,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=train_sampler
    # )
    # print(f"Train Loader Length: {len(train_loader)}")
    # val_dataset = MultiTaskMedicalDataset(val_df, None)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     collate_fn = custom_collate_stratified,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )
    train_pipeline = create_pipeline(train_df, batch_size, device_id=gpu_id, shard_id=0, num_shards=1, 
                    num_threads=1, prefetch_queue_depth=2)
    train_pipeline.build()
    train_loader = DALIGenericIterator(train_pipeline, ["images","seg","labels"], size=-1, auto_reset=False)
    num_train_steps = int(train_df.shape[0]/(batch_size*num_gpu))
    # Initialize model
    model = SwinUNETR(
        in_channels=1,
        out_channels=3,
        feature_size=96
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    checkpoint = None
    if args.use_checkpoint:
        checkpoint = torch.load(args.use_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from checkpoint {args.use_checkpoint}")
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
    model = model.float()
    model = model.to(device)
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)

    #loss function
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        contrastive_weight=args.contrastive_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha
    ).to(device)
    # set up gradscaler
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = np.inf
    start_epoch = args.start_epoch
    for epoch in range(start_epoch, num_epoch):
        # train_sampler.set_epoch(epoch)
        avg_train_loss = train_one_epoch(model,train_loader, criterion, optimizer, scheduler, scaler, args, dist, run=None, epoch=epoch)
        # avg_val_loss = validate(model, val_loader, criterion, epoch, scaler, args, dist, run=None)

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     checkpoint_path = os.path.join(args.output_dir, f"best_model_{epoch}.pth")
        #     if dist.get_rank() == 0:
        #         save_checkpoint(
        #             {'epoch': epoch,
        #              'state_dict': model.module.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict()},
        #             checkpoint_path)
        #         print(f"Best Model saved at {checkpoint_path}")

        # #save the last model 
        # checkpoint_path = os.path.join(args.output_dir, f"last_model.pth")
        # if dist.get_rank() == 0:
        #     save_checkpoint(
        #         {'epoch': epoch,
        #          'state_dict': model.module.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict()},
        #         checkpoint_path)
        #     print(f"Last Model saved at {checkpoint_path}")
        # print(f"Epoch {epoch} completed. Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")
        
    cleanup()
    exit()

if __name__ == '__main__':
    main()


