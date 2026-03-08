import os
import sys
import openml
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def _download_single(ds_id, context, cache_dir):
    """Download and cache one dataset (called from worker threads)."""
    target_name = f"{ds_id}_{context}"
    final_path = os.path.join(cache_dir, "org", "openml", "www", "datasets", target_name)

    if os.path.exists(final_path):
        print(f"  - Already exists: {target_name}")
        return

    try:
        dataset = openml.datasets.get_dataset(ds_id, download_data=True)
        print(f"  - Downloaded: {dataset.name}")

        base_id_path = os.path.join(cache_dir, "org", "openml", "www", "datasets", str(ds_id))
        if os.path.exists(base_id_path):
            os.rename(base_id_path, final_path)
            print(f"  - Renamed to: {target_name}")

    except Exception as e:
        print(f"  - ERROR: Could not process {ds_id}: {e}")


def setup_data():
    """
    Downloads and organises the 20 canonical datasets into the local cache.
    Downloads run in parallel (up to 4 concurrent threads).
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(data_dir, "openml_cache")

    openml.config.set_root_cache_directory(cache_dir)

    print("=" * 70)
    print("EXPLANATION LOTTERY: DATA DOWNLOADER")
    print("=" * 70)
    print(f"Target Directory: {data_dir}")
    print(f"Cache Directory:  {cache_dir}")
    print("-" * 70)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_download_single, ds_id, ctx, cache_dir): ds_id
            for ds_id, ctx in DATASETS.items()
        }
        for future in as_completed(futures):
            ds_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  - ERROR in dataset {ds_id}: {e}")

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE.")
    print("All datasets are organised for the official reproduction pipeline.")
    print("=" * 70)

if __name__ == "__main__":
    setup_data()
