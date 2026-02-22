import os
import sys
import openml
import pandas as pd

# Canonical list of datasets with their context labels
DATASETS = {
    31: "credit",
    37: "diabetes",
    44: "spam",
    50: "tictactoe",
    1046: "software",
    1049: "nasa",
    1050: "nasa",
    1063: "nasa",
    1462: "banknote",
    1464: "blood",
    1479: "hillvalley",
    1480: "liver",
    1494: "biodeg",
    1510: "cancer",
    1590: "census",
    23512: "physics",
    40536: "speeddating",
    40975: "car",
    41027: "junglechess",
    4534: "phishing"
}

def setup_data():
    """
    Downloads and organizes the 20 canonical datasets into the local cache.
    """
    # Fix paths relative to this script's location
    data_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(data_dir, "openml_cache")
    
    # Set OpenML cache directory
    openml.config.set_root_cache_directory(cache_dir)
    
    print("=" * 70)
    print("EXPLANATION LOTTERY: DATA DOWNLOADER")
    print("=" * 70)
    print(f"Target Directory: {data_dir}")
    print(f"Cache Directory:  {cache_dir}")
    print("-" * 70)

    for i, (ds_id, context) in enumerate(DATASETS.items(), 1):
        target_name = f"{ds_id}_{context}"
        # Note: OpenML's internal structure is: root/org/openml/www/datasets/ID
        # But our scripts expect: root/org/openml/www/datasets/ID_context
        # We handle this by set_root_cache_directory and then renaming the leaf folder if needed
        
        print(f"[{i:02d}/20] Checking Dataset {ds_id} ({context})...")
        
        try:
            # Check if finalized folder already exists
            final_path = os.path.join(cache_dir, "org", "openml", "www", "datasets", target_name)
            if os.path.exists(final_path):
                print(f"  - Already exists: {target_name}")
                continue
            
            # Download using OpenML API
            dataset = openml.datasets.get_dataset(ds_id, download_data=True)
            print(f"  - Downloaded: {dataset.name}")
            
            # Rename the folder to include context for our scripts
            base_id_path = os.path.join(cache_dir, "org", "openml", "www", "datasets", str(ds_id))
            if os.path.exists(base_id_path):
                os.rename(base_id_path, final_path)
                print(f"  - Renamed to: {target_name}")
                
        except Exception as e:
            print(f"  - ERROR: Could not process {ds_id}: {e}")

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE.")
    print("All datasets are organized for the official reproduction pipeline.")
    print("=" * 70)

if __name__ == "__main__":
    setup_data()
