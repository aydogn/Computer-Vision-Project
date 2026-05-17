import matplotlib.pyplot as plt
import random
import torch
from dataset import CustomSAMDataset, get_val_test_transforms

def test_dataset():
    try:
        # Use validation or test transform to see deterministic resizing/padding without random augmentations
        dataset = CustomSAMDataset(root_dir="kvasir-seg", split="train", transform=get_val_test_transforms())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the directory structure of the downloaded dataset.")
        return
        
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Pick a random index
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    image = sample['image']
    mask = sample['mask']
    filename = sample['filename']
    
    print(f"Sample index: {idx}")
    print(f"Filename: {filename}")
    print(f"Image shape: {image.shape}, Image min: {image.min():.4f}, max: {image.max():.4f}")
    print(f"Mask shape: {mask.shape}, Mask min: {mask.min()}, max: {mask.max()}")
    print(f"Mask unique values: {torch.unique(mask)}")
    
    # Un-normalize image for visualization
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image_unnorm = image * std + mean
    image_unnorm = torch.clamp(image_unnorm, 0, 1)
    
    # Convert tensors back to numpy for visualization
    # Image: [C, H, W] -> [H, W, C]
    img_np = image_unnorm.permute(1, 2, 0).numpy()
    # Mask: [1, H, W] -> [H, W]
    mask_np = mask.squeeze(0).numpy()
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Image: {filename}\nShape: {img_np.shape}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Binary Mask\nShape: {mask_np.shape}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("test_output_v2.png")
    print("Test visualization saved to 'test_output_v2.png'")

if __name__ == "__main__":
    test_dataset()
