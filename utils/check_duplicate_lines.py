import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the txt file")
    args = parser.parse_args()

    # 用于存储每行内容和对应的行号
    line_dict = defaultdict(list)
    
    # 读取文件并记录行号和内容
    with open(args.file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # 忽略空行
                line_dict[line].append(line_num)
    
    # 检查重复行
    has_duplicates = False
    for line, line_nums in line_dict.items():
        if len(line_nums) > 1:
            has_duplicates = True
            print(f"\n重复行内容: {line}")
            print(f"出现行号: {line_nums}")
            print("-" * 50)
    
    if not has_duplicates:
        print("文件中没有重复的行")

if __name__ == "__main__":
    main() 