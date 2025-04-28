import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the generated .pt file")
    args = parser.parse_args()

    data = torch.load(args.file_path)
    
    print("\nfile content analysis:")
    print("-" * 50)
    
    print("all prompts:")
    for prompt in data.keys():
        print(f"\nPrompt: {prompt}")
    
    print("\ndata shape analysis:")
    for prompt, tensor in data.items():
        print(f"\nPrompt: {prompt}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor device: {tensor.device}")
        
        if len(tensor.shape) == 6:  # [1, 4, num_frames, 16, 60, 104]
            print("\ndim meaning:")
            print(f"dim 0 (batch size): {tensor.shape[0]}")
            print(f"dim 1 (timesteps): {tensor.shape[1]}")
            print(f"dim 2 (video frames): {tensor.shape[2]}")
            print(f"dim 3-5 (spatial dimensions): {tensor.shape[3:]}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()
