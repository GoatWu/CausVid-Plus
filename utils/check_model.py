import torch
import os
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='List all key:value.shape pairs in the generator dictionary of a model file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.pt file')
    args = parser.parse_args()
    
    # Load the model
    model_path = args.model_path
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if 'generator' key exists
        if 'generator' in checkpoint:
            generator = checkpoint['generator']
            
            # Iterate and print all key:value shape pairs
            print("All key:value.shape pairs in the 'generator' dictionary:")
            for key, value in generator.items():
                if hasattr(value, 'shape'):
                    print(f"key: {key}, shape: {value.shape}")
                else:
                    print(f"key: {key}, type: {type(value)}, no shape attribute")
            print(generator["model.blocks.39.self_attn.norm_q.weight"])
            print(generator["model.blocks.39.cross_attn.norm_q.weight"])
        else:
            print("The 'generator' key does not exist in the model")
    else:
        print(f"Model file not found: {model_path}")


if __name__ == "__main__":
    main()