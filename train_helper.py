import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
from pathlib import Path

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, args, dist, run=None, epoch=None):
    model.train()
    total_loss = 0.0
    dice_losses, focal_losses, contrastive_losses = 0.0, 0.0, 0.0
    
    # Synchronize before starting epoch
    if args.nGPU > 1:
        dist.barrier()
        
    if dist.get_rank() == 0:
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    else:
        progress_bar = train_loader 
    for i, data in enumerate(progress_bar):
        if i==10:break  # Note: Remove this in production
        # Clear gradients before forward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # images = images.float().to(args.device, non_blocking=True)  # Use non_blocking
        # labels = labels.float().to(args.device, non_blocking=True)
        # targets = targets.squeeze(1)
        # targets = targets.float().to(args.device, non_blocking=True)

        images = data[0]["images"].float().squeeze(0)  # Access images
        targets = data[0]["seg"].float().squeeze(0)
        targets = targets.squeeze(1)  # Access targets
        labels = data[0]["labels"].float().squeeze(0)  # Access labels
        #print shapes
        print(f"Images shape: {images.shape}, Targets shape: {targets.shape}, Labels shape: {labels.shape}")
        try:
            with torch.cuda.amp.autocast():
                segmentation, logits, feature_maps = model(images, is_training=True)
                loss, dice_loss, focal_loss, contrastive_loss = criterion(segmentation, targets, feature_maps, logits, labels)
            #print the losses
            print(f"Loss: {loss.item()}, Dice Loss: {dice_loss.item()}, Focal Loss: {focal_loss.item()}, Contrastive Loss: {contrastive_loss}")
            # Scale and backward
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            dice_losses += dice_loss.item()
            focal_losses += focal_loss.item()
            contrastive_losses += contrastive_loss

    
            
            if dist.get_rank()==0 and run is not None:
                run["train/loss"].log(loss.item())
                run["train/dice_loss"].log(dice_loss.item())
                run["train/focal_loss"].log(focal_loss.item())
                run["train/contrastive_loss"].log(contrastive_loss.item())
            if dist.get_rank()==0: progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        except RuntimeError as e:
            print(f"Error in training batch {i}: {str(e)}")
            if "out of memory" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            raise e
    

    
    # Synchronize losses across GPUs
    if args.nGPU > 1:
        losses = torch.tensor([total_loss, dice_losses, focal_losses, contrastive_losses], device=args.device)
        dist.all_reduce(losses)
        total_loss, dice_loss, focal_loss, contrastive_loss = losses.tolist()
        total_loss = losses.item()
        total_loss /= dist.get_world_size()
        dice_loss /= dist.get_world_size()
        focal_loss /= dist.get_world_size()
        contrastive_loss /= dist.get_world_size()
        
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

@torch.no_grad()
def validate(model, val_loader, criterion, epoch=None, scaler=None, args=None, dist=None, run=None):
    # Before starting validation
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0.0
    dice_losses, focal_losses, contrastive_losses = 0.0, 0.0, 0.0
    
    # Synchronize before validation
    if args.nGPU > 1:
        dist.barrier()

    if dist.get_rank() == 0:
        progress_bar = tqdm(val_loader, desc=f'valing Epoch {epoch}')
    else:
        progress_bar = val_loader 
    for i, (images, targets, labels) in enumerate(progress_bar):
        # if i==10:break
        try:
            # images = data[0]["images"].float()  # Access images
            # labels = data[0]["labels"].float()  # Access labels
            images = images.float().to(args.device, non_blocking=True)  # Use non_blocking
            labels = labels.float().to(args.device, non_blocking=True)
            targets = targets.squeeze(1)
            targets = targets.float().to(args.device, non_blocking=True)
            with torch.cuda.amp.autocast():
                segmentation, logits, feature_maps = model(images, is_training=True)
                loss, dice_loss, focal_loss, contrastive_loss = criterion(segmentation, targets, feature_maps, logits, labels)
            total_loss += loss.item()
            dice_losses += dice_loss.item()
            focal_losses += focal_loss.item()
            contrastive_losses += contrastive_loss
            
            if dist.get_rank()==0 and run is not None:
                run["val/loss"].log(loss.item())
                run["val/dice_loss"].log(dice_loss.item())
                run["val/focal_loss"].log(focal_loss.item())
                run["val/contrastive_loss"].log(contrastive_loss.item())
            if dist.get_rank()==0: progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                        
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            raise e
    
    # Synchronize losses across GPUs
    if args.nGPU > 1:
        losses = torch.tensor([total_loss, dice_losses, focal_losses, contrastive_losses], device=args.device)
        dist.all_reduce(losses)
        total_loss, dice_loss, focal_loss, contrastive_loss = losses.tolist()
        total_loss = losses.item()
        total_loss /= dist.get_world_size()
        dice_loss /= dist.get_world_size()
        focal_loss /= dist.get_world_size()
        contrastive_loss /= dist.get_world_size()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

