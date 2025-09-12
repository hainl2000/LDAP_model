#!/usr/bin/env python3
import numpy as np
import os

def inspect_npy_files(dataset="dataset1"):
    """Inspect the structure and content of .npy files"""
    
    base_path = f"./our_dataset/{dataset}/index/"
    
    print(f"=== Inspecting {dataset} ===\n")
    
    # Check if files exist
    files_to_check = [
        "positive_ij.npy",
        "negative_ij.npy", 
        "positive5foldsidx.npy",
        "negative5foldsidx.npy"
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            print(f"❌ {filename} not found!")
            continue
            
        print(f"📁 {filename}:")
        
        try:
            if "5foldsidx" in filename:
                # These contain Python objects
                data = np.load(filepath, allow_pickle=True)
                print(f"  Type: {type(data)}")
                print(f"  Shape: {data.shape}")
                
                # If it's a 0-d array containing a dict/list
                if data.ndim == 0:
                    content = data.item()
                    print(f"  Content type: {type(content)}")
                    if isinstance(content, dict):
                        print(f"  Keys: {list(content.keys())}")
                        if 0 in content:
                            print(f"  Fold 0 keys: {list(content[0].keys())}")
                            print(f"  Fold 0 train size: {len(content[0]['train'])}")
                            print(f"  Fold 0 test size: {len(content[0]['test'])}")
                else:
                    print(f"  First element type: {type(data[0])}")
                    
            else:
                # Regular numpy arrays
                data = np.load(filepath)
                # print(f"  Shape: {data.shape}")
                # print(f"  Dtype: {data.dtype}")
                # print(f"  Min/Max: {data.min()} / {data.max()}")
                # print(f"  First 3 rows:")
                print(f"    {data[:10]}")
                
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
            
        print()

if __name__ == "__main__":
    # Inspect all datasets
    for dataset in ["dataset1", "dataset2", "dataset3"]:
        if os.path.exists(f"./our_dataset/{dataset}"):
            inspect_npy_files(dataset)
            print("-" * 50)
