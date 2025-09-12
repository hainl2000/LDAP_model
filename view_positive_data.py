#!/usr/bin/env python3
import numpy as np
import os

def view_positive_ij_details(dataset="dataset1"):
    """View detailed row data of positive_ij.npy"""
    
    filepath = f"./our_dataset/{dataset}/index/positive_ij.npy"
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"=== Detailed view of positive_ij.npy from {dataset} ===\n")
    
    try:
        # Load the data
        positive_ij = np.load(filepath)
        
        # Basic information
        print(f"📊 Basic Information:")
        print(f"  Shape: {positive_ij.shape}")
        print(f"  Data type: {positive_ij.dtype}")
        print(f"  Total elements: {positive_ij.size}")
        print(f"  Memory usage: {positive_ij.nbytes} bytes")
        print()
        
        # Statistical information
        print(f"📈 Statistical Information:")
        if positive_ij.ndim == 2:
            print(f"  Min values per column: {positive_ij.min(axis=0)}")
            print(f"  Max values per column: {positive_ij.max(axis=0)}")
            print(f"  Mean values per column: {positive_ij.mean(axis=0)}")
        else:
            print(f"  Min value: {positive_ij.min()}")
            print(f"  Max value: {positive_ij.max()}")
            print(f"  Mean value: {positive_ij.mean()}")
        print()
        
        # Show all data rows
        print(f"📋 All Row Data:")
        print(f"  Row format: [index] -> values")
        print("-" * 50)
        
        if positive_ij.ndim == 1:
            # 1D array
            for i, value in enumerate(positive_ij):
                print(f"  [{i:3d}] -> {value}")
        elif positive_ij.ndim == 2:
            # 2D array
            for i, row in enumerate(positive_ij):
                if positive_ij.shape[1] <= 10:
                    # Show all columns if not too many
                    print(f"  [{i:3d}] -> {row}")
                else:
                    # Show first and last few columns if too many
                    print(f"  [{i:3d}] -> [{row[0]}, {row[1]}, ..., {row[-2]}, {row[-1]}]")
                    
                # Show first 20 rows, then ask if user wants to see more
                if i >= 19 and i < len(positive_ij) - 1:
                    remaining = len(positive_ij) - i - 1
                    print(f"  ... ({remaining} more rows)")
                    break
        else:
            print(f"  Data has {positive_ij.ndim} dimensions - showing flattened view:")
            flat_data = positive_ij.flatten()
            for i, value in enumerate(flat_data[:50]):  # Show first 50 elements
                print(f"  [{i:3d}] -> {value}")
            if len(flat_data) > 50:
                print(f"  ... ({len(flat_data) - 50} more elements)")
        
        print("-" * 50)
        print(f"✅ Total rows displayed: {min(20, len(positive_ij)) if positive_ij.ndim >= 1 else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    # Check which datasets are available
    available_datasets = []
    for dataset in ["dataset1", "dataset2", "dataset3"]:
        if os.path.exists(f"./our_dataset/{dataset}"):
            available_datasets.append(dataset)
    
    print(f"Available datasets: {available_datasets}\n")
    
    # View positive_ij for all available datasets
    for dataset in available_datasets:
        view_positive_ij_details(dataset)
        print("\n" + "="*60 + "\n")
