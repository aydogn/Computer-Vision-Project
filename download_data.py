import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

def download_kvasir_seg():
    print("Downloading Kvasir-SEG dataset from Hugging Face...")
    local_dir = "kvasir-seg"
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the dataset repository
    snapshot_download(
        repo_id="Angelou0516/kvasir-seg",
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=4,
    )
    print(f"Dataset downloaded successfully to {local_dir}")

if __name__ == "__main__":
    download_kvasir_seg()
