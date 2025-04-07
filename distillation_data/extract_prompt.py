import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract keys from JSON file and write to TXT file')
    parser.add_argument('--json_path', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--txt_path', type=str, required=True, help='Path to output TXT file')
    
    args = parser.parse_args()
    
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    keys = list(data.keys())
    
    with open(args.txt_path, 'w') as f:
        for key in keys:
            f.write(f"{key}\n")
    
    print(f"Successfully extracted {len(keys)} keys to {args.txt_path}")

if __name__ == "__main__":
    main() 