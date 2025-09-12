#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd # Using pandas for more robust CSV saving, especially for object arrays

def convert_npy_to_csv(input_dir="./our_dataset", output_dir="./our_csv_dataset"):
    """Converts all .npy files in input_dir to .csv files in output_dir."""

    print(f"Starting conversion from {input_dir} to {output_dir}\n")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    npy_files_found = 0
    csv_files_created = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_files_found += 1
                npy_filepath = os.path.join(root, file)
                
                # Construct the corresponding output path
                relative_path = os.path.relpath(npy_filepath, input_dir)
                csv_filename = file.replace(".npy", ".csv")
                csv_filepath = os.path.join(output_dir, relative_path.replace(file, csv_filename))

                # Create necessary subdirectories in the output path
                os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

                print(f"Converting: {npy_filepath} -> {csv_filepath}")
                
                try:
                    # Try loading with allow_pickle=False first for regular arrays
                    try:
                        data = np.load(npy_filepath)
                    except ValueError as e:
                        if "pickle" in str(e):
                            # If it's a pickle error, try with allow_pickle=True
                            data = np.load(npy_filepath, allow_pickle=True)
                        else:
                            raise # Re-raise other ValueErrors

                    # Convert to DataFrame for easier CSV writing, especially for object arrays
                    if isinstance(data, np.ndarray):
                        if data.dtype == 'object':
                            # Handle object arrays (e.g., arrays of dictionaries)
                            # This might require custom serialization or just saving as string
                            # For now, convert to a list of strings if it's not directly convertible
                            if data.ndim == 0:
                                # Scalar object array
                                df = pd.DataFrame([data.item()])
                            elif data.ndim == 1:
                                # 1D object array
                                df = pd.DataFrame(data)
                            else:
                                # Multi-dimensional object array - flatten or handle carefully
                                print(f"  Warning: Multi-dimensional object array detected. Saving as string representation: {npy_filepath}")
                                df = pd.DataFrame(data.astype(str))
                        else:
                            # Regular numeric or boolean arrays
                            df = pd.DataFrame(data)
                    elif isinstance(data, (list, dict, int, float, str, bool)):
                        # Handle cases where np.load returns a scalar or non-array object
                        df = pd.DataFrame([data])
                    else:
                        print(f"  Warning: Unexpected data type after loading {npy_filepath}: {type(data)}. Attempting to convert to DataFrame.")
                        df = pd.DataFrame(data)

                    df.to_csv(csv_filepath, index=False, header=False) # No index, no header
                    csv_files_created += 1
                    print(f"  ✅ Converted successfully.")

                except Exception as e:
                    print(f"  ❌ Failed to convert {npy_filepath}: {e}")
                print("-" * 30)

    print(f"\nConversion finished.")
    print(f"Found {npy_files_found} .npy files.")
    print(f"Created {csv_files_created} .csv files.")

if __name__ == "__main__":
    convert_npy_to_csv()
