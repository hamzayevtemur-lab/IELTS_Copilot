from datasets import load_dataset
import os

def download_dataset():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Fetching dataset 'chillies/ielts-writing-task2-essays' from Hugging Face...")
    ds = load_dataset("chillies/ielts-writing-task2-essays")
    
    print("\nDataset structure:")
    print(ds)
    
    # Export splits (e.g., 'train') to CSV and Parquet
    for split in ds.keys():
        csv_path = os.path.join(data_dir, f"ielts_task2_{split}.csv")
        parquet_path = os.path.join(data_dir, f"ielts_task2_{split}.parquet")
        
        print(f"\nSaving '{split}' split ({len(ds[split])} rows)...")
        ds[split].to_csv(csv_path, index=False)
        ds[split].to_parquet(parquet_path)
        print(f" -> Saved CSV to: {csv_path}")
        print(f" -> Saved Parquet to: {parquet_path}")

    # Also save in native HuggingFace dataset format
    hf_save_dir = os.path.join(data_dir, "ielts_writing_task2_hf")
    ds.save_to_disk(hf_save_dir)
    print(f"\n -> Saved Arrow dataset to: {hf_save_dir}")
    print("\nDownload complete!")

if __name__ == "__main__":
    download_dataset()
