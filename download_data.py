from huggingface_hub import snapshot_download
import os

def download_kvasir_seg():
    print("Downloading Kvasir-SEG dataset from Hugging Face...")
    local_dir = "kvasir-seg"
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the dataset repository
    snapshot_download(
        repo_id="Angelou0516/kvasir-seg",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Dataset downloaded successfully to {local_dir}")

if __name__ == "__main__":
    download_kvasir_seg()
