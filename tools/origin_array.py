import json
import re

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_string(s):
    patterns = ["a photo of an ", "a photo of a ", "a photo of one "]
    for pattern in patterns:
        s = re.sub(pattern, "", s)
    return s

def generate_repeated_keys(data):
    result = {}
    for key in data.keys():
        cleaned_key = clean_string(key)
        result[cleaned_key] = [clean_string(key)] * 10
    return result

def write_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 主函数
def main(input_file, output_file):
    data = read_json_file(input_file)
    repeated_keys_data = generate_repeated_keys(data)
    write_json_file(repeated_keys_data, output_file)
    print(f"JSON with repeated keys has been saved to {output_file}")

# 使用示例
input_file = './debias/data/truth.json'  # 输入的JSON文件
output_file = './debias/data/origin_array.json'  # 输出的JSON文件
main(input_file, output_file)

