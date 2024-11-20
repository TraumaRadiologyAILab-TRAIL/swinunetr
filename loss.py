import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MultiScaleMaskGuidedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, scale_weights=[1.0, 0.8, 0.6, 0.4, 0.2]):
        super().__init__()
        self.temperature = temperature
        self.scale_weights = scale_weights

    def forward(self, feature_maps, masks):
        """
        Args:
            feature_maps: List of feature maps at different scales
                        [B, C, H, W, D] for each scale
            masks: Original segmentation masks [B, 1, H, W, D]
        """
        total_loss = 0
        
        for scale_idx, (features, weight) in enumerate(zip(feature_maps, self.scale_weights)):
            # Resize mask to match current feature scale
            curr_size = features.shape[2:]
            scaled_masks = F.interpolate(
                masks.float(), 
                size=curr_size,
                mode='nearest'
            ).long()
            
            scale_loss = self.compute_scale_loss(features, scaled_masks)
            total_loss += weight * scale_loss
            
        return total_loss

    def compute_scale_loss(self, features, masks):
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        
        batch_loss = 0
        for b in range(batch_size):
            curr_feat = features[b]  # [C, H, W, D]
            curr_mask = masks[b, 0]  # [H, W, D]
            
            # Extract features for each class
            htx_feat = curr_feat[:, curr_mask == 1]  
            mh_feat = curr_feat[:, curr_mask == 2]   
            bg_feat = curr_feat[:, curr_mask == 0]   
            
            loss = 0
            
            # Intra-class (positive) similarities
            if htx_feat.shape[1] > 0:
                htx_sim = self.compute_similarity(htx_feat)
                loss += self.positive_loss(htx_sim)
            
            if mh_feat.shape[1] > 0:
                mh_sim = self.compute_similarity(mh_feat)
                loss += self.positive_loss(mh_sim)
            
            # Inter-class (negative) similarities
            if htx_feat.shape[1] > 0 and mh_feat.shape[1] > 0:
                diff_sim = self.compute_similarity(htx_feat, mh_feat)
                loss += self.negative_loss(diff_sim)
            
            # Background contrasting
            if bg_feat.shape[1] > 0:
                if htx_feat.shape[1] > 0:
                    bg_htx_sim = self.compute_similarity(bg_feat, htx_feat)
                    loss += self.negative_loss(bg_htx_sim)
                
                if mh_feat.shape[1] > 0:
                    bg_mh_sim = self.compute_similarity(bg_feat, mh_feat)
                    loss += self.negative_loss(bg_mh_sim)
            
            batch_loss += loss
            
        return batch_loss / batch_size

    def compute_similarity(self, feat1, feat2=None):
        """Compute cosine similarity"""
        if feat2 is None:
            return torch.matmul(feat1.T, feat1)
        return torch.matmul(feat1.T, feat2)

    def positive_loss(self, similarity):
        similarity = similarity / self.temperature
        mask = ~torch.eye(similarity.shape[0], device=similarity.device).bool()
        similarity = similarity[mask].reshape(similarity.shape[0], -1)
        return -torch.log(torch.exp(similarity).mean() + 1e-6)

    def negative_loss(self, similarity):
        similarity = similarity / self.temperature
        return torch.log(1 + torch.exp(similarity).mean())



class MulticlassDiceLoss(nn.Module):
    def __init__(self, include_background=False, smooth=1e-5, weight=None):
        """
        Args:
            include_background (bool): Whether to include the background class in the Dice loss computation.
            smooth (float): Smoothing factor to avoid division by zero.
            weight (list or None): Class weights. If specified, it should have one weight per class.
        """
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits of shape [B, C, H, W, D] where C is the number of classes.
            target (torch.Tensor): Ground truth of shape [B, H, W, D] with integer class labels (0 to C-1).

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        # Ensure predictions are softmax-normalized probabilities
        pred = F.softmax(pred, dim=1)

        # Convert target to one-hot encoding
        print(f"Target min: {target.min()}, Target max: {target.max()}")
        target[target == -1] = 0
        target = target.long()
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        # Optionally exclude background class
        start_class = 0 if self.include_background else 1

        dice_scores = []
        for class_idx in range(start_class, num_classes):
            pred_class = pred[:, class_idx]  # [B, H, W, D]
            # print(pred_class.shape) # [B, H, W, D]
            target_class = target_one_hot[:, class_idx]  # [B, H, W, D]
            # print(target_class.shape)
            # Compute intersection and union
            intersection = (pred_class * target_class).sum(dim=(1, 2, 3))
            union = pred_class.sum(dim=(1, 2, 3)) + target_class.sum(dim=(1, 2, 3))

            # Compute Dice score
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            # Apply class weight if specified
            if self.weight is not None:
                dice = dice * self.weight[class_idx]

            dice_scores.append(dice)

        # If no classes are included (e.g., all masks are empty), return zero loss
        if len(dice_scores) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Compute mean Dice score across classes and batches
        dice_loss = 1 - torch.cat(dice_scores).mean()
        return dice_loss

# Usage example with logging
class DiceMetric:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.htx_dices = []
        self.mh_dices = []
        self.bg_dices = []
        
    def update(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=3).permute(0, 4, 1, 2, 3)
        
        # Calculate Dice for each class
        for class_idx, class_name in enumerate(['bg', 'htx', 'mh']):
            pred_class = pred[:, class_idx]
            target_class = target[:, class_idx]
            
            if target_class.sum() == 0:
                continue
                
            intersection = (pred_class * target_class).sum(dim=(1, 2, 3))
            union = pred_class.sum(dim=(1, 2, 3)) + target_class.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection) / (union + 1e-5)
            
            if class_name == 'bg':
                self.bg_dices.extend(dice.cpu().numpy())
            elif class_name == 'htx':
                self.htx_dices.extend(dice.cpu().numpy())
            else:  # mh
                self.mh_dices.extend(dice.cpu().numpy())
    
    def compute(self):
        metrics = {}
        if self.bg_dices:
            metrics['bg_dice'] = np.mean(self.bg_dices)
        if self.htx_dices:
            metrics['htx_dice'] = np.mean(self.htx_dices)
        if self.mh_dices:
            metrics['mh_dice'] = np.mean(self.mh_dices)
        return metrics

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        
        pt = torch.sigmoid(pred)
        pt = torch.where(target == 1, pt, 1 - pt)
        
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class CombinedLoss(nn.Module):
   def __init__(
       self,
       num_classes=3,
       dice_weight=1.0,
       focal_weight=1.0,
       contrastive_weight=0.1,
       focal_gamma=2.0,
       focal_alpha=0.25,
       temperature=0.1,
       scale_weights=[1.0, 0.8, 0.6, 0.4, 0.2],
       smooth=1e-5
   ):
       super().__init__()
       self.num_classes = num_classes
       self.dice_weight = dice_weight
       self.focal_weight = focal_weight
       self.contrastive_weight = contrastive_weight
       
       # Initialize individual losses
       self.dice_loss = MulticlassDiceLoss(
           include_background=False,
           smooth=smooth
       )
       
       self.focal_loss = FocalLoss(
           gamma=focal_gamma,
           alpha=focal_alpha,
           reduction='mean'
       )
       
       self.contrastive_loss = MultiScaleMaskGuidedContrastiveLoss(
           temperature=temperature,
           scale_weights=scale_weights
       )
       
   def forward(self, pred_maps, target, feature_maps, logits, labels):
       """
       Args:
           pred_maps: Final segmentation output [B, C, H, W, D]
           target: Ground truth masks [B, H, W, D] with values in {0,1,2}
           feature_maps: List of intermediate feature maps for contrastive loss
                       Each with shape [B, C, H', W', D']
       """
       # Expand target dimensions for focal loss
       
       # Calculate individual losses
    #    print(target.shape)
       dice_loss = self.dice_loss(pred_maps, target)
       focal_loss = self.focal_loss(logits, labels)
    #    print(target.shape)
       contrastive_loss = self.contrastive_loss(feature_maps, target.unsqueeze(1))
       
       # Combine losses
       total_loss = (
           self.dice_weight * dice_loss +
           self.focal_weight * focal_loss +
           self.contrastive_weight * contrastive_loss
       )
       
       return total_loss, dice_loss, focal_loss, contrastive_loss
    


    