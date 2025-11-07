MAX_ID = 1822

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"

import os
import requests
import time
import argparse
from multiprocessing import Pool
import pyarrow.parquet as pq

DATA_DIR = 'minichat/data'
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(index):
    filename = f'shard_{index:05d}.parquet'
    filepath = os.path.join(DATA_DIR, filename) 
    # Skip if file already exists
    if os.path.exists(filepath):
        print(f"File {filename} already exists, skipping...")

    url = f'{BASE_URL}/{filename}'
    temp_filepath = filepath + '.tmp'
    
    # Retry download up to 5 times with exponential backoff
    for attempt in range(5):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            os.rename(temp_filepath, filepath)
            print(f"Successfully downloaded {filename}")
            return True
            
            
        except (requests.RequestException, OSError) as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
            print(f"Attempt {attempt + 1} failed for {filename}: {e}")
            
            if attempt == 4:  # Last attempt
                print(f"Failed to download {filename} after 5 attempts")
                # Clean up temp file if it exists
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            else:
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        return False

def list_files(data_dir=DATA_DIR):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]


def iter_batched(split, start=0, step=1):
    assert split in ['train', 'val'], f"Invalid split: {split}"
    all_files = list_files()
    paths = all_files[:-1] if split == 'train' else all_files[-1:]
    for file_path in paths:
        pf = pq.ParquetFile(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download dataset shards")
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('-w', '--num-files', type=int, default=MAX_ID, help='Number of files to download')
    args = parser.parse_args()

    ids_to_download = list(range(args.num_files))

    with Pool(args.num_workers) as pool:
        results = pool.map(download_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Successfully downloaded {successful}/{args.num_files} files.")
    
