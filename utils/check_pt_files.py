import torch
import glob
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing .pt files")
    args = parser.parse_args()

    pt_files = sorted(glob.glob(os.path.join(args.folder_path, "*.pt")))
    
    print(f"Found {len(pt_files)} .pt files")
    print("-" * 50)
    
    expected_shape = torch.Size([1, 4, 21, 16, 60, 104])
    
    for file_path in tqdm(pt_files, desc="Checking files"):
        try:
            data = torch.load(file_path)
            
            for prompt, tensor in data.items():
                if tensor.shape != expected_shape:
                    print(f"\nFile: {os.path.basename(file_path)}")
                    print(f"Prompt: {prompt}")
                    print(f"Expected shape: {expected_shape}")
                    print(f"Actual shape: {tensor.shape}")
                    print("-" * 30)
                
        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main() 