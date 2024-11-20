import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.pickling as dali_pickle
from nvidia.dali import pipeline_def, fn
from collections import defaultdict
from nvidia.dali.auto_aug import rand_augment
from PIL import Image
from pathlib import Path
from monai.transforms import (
    Resized,
    GridPatchd
)
from monai.data import MetaTensor 

def get_patch_label(patch):
    """Generate label for a single patch based on mask values"""
    unique_values = np.unique(patch)
    has_one = 1 in unique_values
    has_two = 2 in unique_values
    
    if has_one and has_two:
        return torch.tensor([1, 1], dtype=torch.float32)
    elif has_one:
        return torch.tensor([1, 0], dtype=torch.float32)
    elif has_two:
        return torch.tensor([0, 1], dtype=torch.float32)
    return torch.tensor([0, 0], dtype=torch.float32)
def patchify(file):
    """Divide image and mask into patches and generate labels for each patch"""
    patchify = GridPatchd(keys=["data", "seg"], patch_size=(128, 128, 128))
    patchified_file = patchify(file)
    
    # Get patches
    image_patches = patchified_file['data']
    mask_patches = patchified_file['seg']
    
    # Generate labels for each patch
    patch_labels = []
    for patch in mask_patches:
        label = get_patch_label(patch)
        patch_labels.append(label)
    
    # Stack all labels into a single tensor
    patch_labels = torch.stack(patch_labels)
    # if patch_labels.sum() > 0:
    #     print(f"Found positive label: {patch_labels.sum()}")
    # print(f"Number of patches: {len(image_patches)}")
    # print(f"Patch labels shape: {patch_labels.shape}")
    
    return image_patches, mask_patches, patch_labels


class PatchManager:
    def __init__(self, df, batch_size=4, cases_per_load=50):
        self.df = df
        self.batch_size = batch_size
        self.cases_per_load = cases_per_load
        self.current_batch = {"00": [], "10": [], "01": [], "11": []}
        self.case_index = 0
        
    def load_next_batch(self):
        self.current_batch = {"00": [], "10": [], "01": [], "11": []}
        end_idx = min(self.case_index + self.cases_per_load, len(self.df))
        cases = self.df.iloc[self.case_index:end_idx]
        
        for _, row in cases.iterrows():
            try:
                files = np.load(f"/storage/data/nnunet/nnUNet_Plans_3d_fullres/{row.case_id}.npz")
                image_patches, mask_patches, patch_labels = patchify(files)
                
                for i, (img, mask, label) in enumerate(zip(image_patches, mask_patches, patch_labels)):
                    cat = "".join(map(str, label.int().tolist()))
                    self.current_batch[cat].append({
                        "image": img,
                        "mask": mask,
                        "label": label
                    })
            except Exception as e:
                print(f"Error processing case {row.case_id}: {e}")
                continue
        
        self.case_index = end_idx
        if self.case_index >= len(self.df):
            self.case_index = 0
            
    def create_balanced_batch(self):
        # Check if current batch is depleted
        total_patches = sum(len(patches) for patches in self.current_batch.values())
        if total_patches < self.batch_size:
            self.load_next_batch()
        
        batch_images = []
        batch_masks = []
        batch_labels = []
        
        # Try to get one sample from each category
        categories = list(self.current_batch.keys())
        np.random.shuffle(categories)
        
        for cat in categories:
            if self.current_batch[cat]:
                idx = np.random.randint(len(self.current_batch[cat]))
                patch = self.current_batch[cat].pop(idx)
                batch_images.append(patch["image"])
                batch_masks.append(patch["mask"])
                batch_labels.append(patch["label"])
        
        # Fill remaining slots randomly from non-empty categories
        while len(batch_images) < self.batch_size:
            valid_cats = [cat for cat in self.current_batch if self.current_batch[cat]]
            if not valid_cats:
                self.load_next_batch()
                valid_cats = [cat for cat in self.current_batch if self.current_batch[cat]]
            
            cat = np.random.choice(valid_cats)
            idx = np.random.randint(len(self.current_batch[cat]))
            patch = self.current_batch[cat].pop(idx)
            batch_images.append(patch["image"])
            batch_masks.append(patch["mask"])
            batch_labels.append(patch["label"])
        
        return (torch.stack(batch_images), 
                torch.stack(batch_masks), 
                torch.stack(batch_labels))


@dali_pickle.pickle_by_value
def create_callback(df, batch_size, shard_id=0, num_shards=1, seed=42):
    patch_manager = PatchManager(df, batch_size=4)
    
    def callback(sample_info):
        try:
            batch_images, batch_masks, batch_labels = patch_manager.create_balanced_batch()
            return batch_images.as_tensor(), batch_masks.as_tensor(), batch_labels
            
        except Exception as e:
            print(f"Error creating batch: {e}")
            return (torch.zeros((batch_size, 1, 128, 128, 128)),
                   torch.zeros((batch_size, 1, 128, 128, 128)),
                   torch.zeros((batch_size, 2)))
    
    return callback

def create_pipeline(df, batch_size, device_id, shard_id=0, num_shards=1, 
                   num_threads=16, prefetch_queue_depth=2, seed=42):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, 
                   device_id=device_id, py_start_method="spawn")
    with pipe:
        x, y, z = fn.external_source(
            source=create_callback(df, batch_size, shard_id, num_shards, seed),
            num_outputs=3,
            batch=False,
            parallel=True,
            prefetch_queue_depth=prefetch_queue_depth
        )
        
        x = fn.cast(x, dtype=types.FLOAT)
        y = fn.cast(y, dtype=types.FLOAT)
        z = fn.cast(z, dtype=types.FLOAT)
        pipe.set_outputs(x.gpu(), y.gpu(), z.gpu())
    
    return pipe