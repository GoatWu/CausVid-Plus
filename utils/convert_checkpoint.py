import torch
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and save the generator part from a checkpoint.')
    parser.add_argument('--input-checkpoint', type=str, required=True, help='Path to the input checkpoint file')
    parser.add_argument('--output-checkpoint', type=str, required=True, help='Path to save the output checkpoint file')
    parser.add_argument('--remove-prefix', type=str, default="model.", help='Prefix to remove from keys (default: "model.")')
    parser.add_argument('--to-bf16', action='store_true', help='Convert model weights to bfloat16')
    args = parser.parse_args()
    
    # Extract arguments
    input_path = args.input_checkpoint
    output_path = args.output_checkpoint
    prefix_to_remove = args.remove_prefix
    convert_to_bf16 = args.to_bf16
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input checkpoint file not found: {input_path}")
        return
    
    # Load the input checkpoint
    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
    
    # Check if 'generator' key exists
    if 'generator' not in checkpoint:
        print("Error: The 'generator' key does not exist in the input checkpoint")
        return
    
    # Extract the generator
    generator = checkpoint['generator']
    print(f"Successfully extracted 'generator' from input checkpoint")
    
    # Remove the specified prefix from keys
    new_generator = {}
    prefix_count = 0
    tensor_count = 0
    
    for key, value in generator.items():
        # Process key - remove prefix if needed
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]  # Remove the prefix
            prefix_count += 1
        else:
            new_key = key
        
        # Convert tensor to bf16 if requested
        if convert_to_bf16 and isinstance(value, torch.Tensor) and value.is_floating_point():
            value = value.to(torch.bfloat16)
            tensor_count += 1
        
        new_generator[new_key] = value
    
    # Print processing summary
    print(f"Removed prefix '{prefix_to_remove}' from {prefix_count} keys")
    if convert_to_bf16:
        print(f"Converted {tensor_count} tensors to bfloat16")
    
    # Save the new checkpoint
    print(f"Saving generator to {output_path}...")
    torch.save(new_generator, output_path)
    print(f"Successfully saved generator to {output_path}")

if __name__ == "__main__":
    main()
