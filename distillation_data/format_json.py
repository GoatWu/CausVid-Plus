#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

def format_json_file(input_path: str, output_path: str = None, indent: int = 2):
    """
    Format single-line JSON file into multiple lines with one key-value pair per line
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output file (if None, will overwrite input file)
        indent: Number of spaces for indentation
    """
    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # If no output path specified, use input path
        if not output_path:
            output_path = input_path
            
        # Write formatted JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            
        print(f"Successfully formatted JSON file: {output_path}")
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in input file - {e}")
    except IOError as e:
        print(f"Error: Failed to read/write file - {e}")
    except Exception as e:
        print(f"Error: Unexpected error occurred - {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Format single-line JSON file into multiple lines'
    )
    
    parser.add_argument(
        'input',
        help='Input JSON file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path (optional, will overwrite input if not specified)',
        default=None
    )
    
    parser.add_argument(
        '-i', '--indent',
        help='Number of spaces for indentation (default: 2)',
        type=int,
        default=2
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found - {args.input}")
        return
        
    # Format JSON file
    format_json_file(args.input, args.output, args.indent)

if __name__ == "__main__":
    main()
