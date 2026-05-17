import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Albumentations transforms for the training set.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=1024, 
            min_width=1024, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_test_transforms():
    """
    Albumentations transforms for the validation and test sets (no augmentations).
    """
    return A.Compose([
        A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=1024, 
            min_width=1024, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class CustomSAMDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Root directory of the Kvasir-SEG dataset.
            split (string): 'train', 'validation', or 'test'.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.transform = transform
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Image or mask directory not found for split '{split}' in {root_dir}")
            
        # Filter files to only include those that have both image and mask
        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.exists(os.path.join(self.mask_dir, f)) or 
               os.path.exists(os.path.join(self.mask_dir, os.path.splitext(f)[0] + '.png')) or
               os.path.exists(os.path.join(self.mask_dir, os.path.splitext(f)[0] + '.jpg'))
        ])
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        mask_path_jpg = os.path.join(self.mask_dir, img_name)
        mask_path_png = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + '.png')
        
        if os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        elif os.path.exists(mask_path_png):
            mask_path = mask_path_png
        else:
            raise FileNotFoundError(f"Mask for {img_name} not found.")

        # Read image using OpenCV and convert BGR -> RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask at {mask_path}")
        
        # Binarize mask before Albumentations to guarantee 0 and 1 values
        mask = (mask > 127).astype(np.float32)
        
        # Apply Albumentations transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Albumentations ToTensorV2 returns 2D masks as [H, W], but we need [1, H, W] for the model
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        sample = {'image': image, 'mask': mask, 'filename': img_name}
        return sample

def get_dataloaders(root_dir="kvasir-seg", batch_size=2):
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test sets.
    """
    train_dataset = CustomSAMDataset(root_dir, split="train", transform=get_train_transforms())
    val_dataset = CustomSAMDataset(root_dir, split="validation", transform=get_val_test_transforms())
    test_dataset = CustomSAMDataset(root_dir, split="test", transform=get_val_test_transforms())
    
    # shuffle=True for train, shuffle=False for validation and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
