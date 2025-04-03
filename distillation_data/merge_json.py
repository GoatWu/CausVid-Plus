#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Merge two video annotation JSON files')
    parser.add_argument('--input1', type=str, required=True, help='Path to the first input JSON file')
    parser.add_argument('--input2', type=str, required=True, help='Path to the second input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output merged JSON file')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Loading first file: {args.input1}")
    try:
        with open(args.input1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
    except Exception as e:
        print(f"Error reading the first file: {str(e)}")
        return

    print(f"Loading second file: {args.input2}")
    try:
        with open(args.input2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except Exception as e:
        print(f"Error reading the second file: {str(e)}")
        return
    
    # Check file types
    if not isinstance(data1, list) or not isinstance(data2, list):
        print("Error: Both files should contain JSON arrays")
        return
    
    # Simply merge the two lists
    merged_data = data1 + data2
    print(f"Merged {len(data1)} and {len(data2)} records, total {len(merged_data)} records")
    
    # Write merged data to output file
    print(f"Writing merged data to: {args.output}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"Merge completed! Results saved to: {args.output}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
        return

if __name__ == "__main__":
    main() 