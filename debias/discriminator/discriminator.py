import json
from scipy.spatial.distance import cosine
import random
import os

#functions
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    file.close()
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    file.close()

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    file.close()
    return lines

def write_txt_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    file.close()

def modify_json_keys(data):
    modified_data = {}
    for key, value in data.items():
        lower_key = key.lower()  
        if lower_key.startswith("a photo of one "):
            new_key = key[len("a photo of one "):]
        elif lower_key.startswith("a photo of a "):
            new_key = key[len("a photo of a "):]
        elif lower_key.startswith("a photo of an "):
            new_key = key[len("a photo of an "):]
        else:
            new_key = key
        modified_data[new_key] = value
    return modified_data

def do_difference(data_demographic, data_generative):
    result = {}
    for key in data_demographic.keys():
        if key in data_generative:
            result[key] = {}
            for sub_key in data_demographic[key]:
                if sub_key in data_generative[key]:
                    result[key][sub_key] = {}
                    for sub_sub_key in data_demographic[key][sub_key]:
                        if sub_sub_key in data_generative[key][sub_key]:
                            result[key][sub_key][sub_sub_key] = (
                                data_generative[key][sub_key][sub_sub_key] - data_demographic[key][sub_key][sub_sub_key]
                            )
    return result

def get_array(json_data):
    result = {}
    for key in json_data:
        if 'array' in json_data[key]:
            result[key] = json_data[key]['array']
    return result

def modify_array(data_array, data_modifications):
    for key in data_modifications:
        if key in data_array:
            modifications = []
            for sub_key in data_modifications[key]:
                for sub_sub_key, value in data_modifications[key][sub_key].items():
                    if value < 0:
                        modifications.append((sub_sub_key, sub_key))
            for modification in modifications:
                sub_sub_key, sub_key = modification
                for i in range(len(data_array[key])):
                    if data_array[key][i] == key:
                        data_array[key][i] = f"{sub_sub_key} {key}"
                        break
    return data_array

def process_list(strings):
    processed_list = []
    for s in strings:
        s = s.replace("0-30", "young")
        s = s.replace("30-60", "middle-aged")
        s = s.replace("60+", "elderly")
        if "person" in s and "female " in s:
            s = s.replace("female ", "").replace("person", "woman")
        elif "person" in s and "male " in s:
            s = s.replace("male ", "").replace("person", "man")
        s = s.strip()
        processed_list.append(s)
    return processed_list

def process_json(data):
    for key in data.keys():
        data[key] = process_list(data[key])
    return data

def process_prompts(json_data, txt_lines):
    processed_lines = []
    for line in txt_lines:
        if line.strip().startswith("--prompt"):
            parts = line.split(",", 1)
            first_part = parts[0].split(" ", 3)[-1]
            for key in json_data:
                if key in first_part:
                    words = first_part.split(" ")
                    suffix = ""
                    for index in range(2, len(words)):
                        suffix = suffix + words[index] + " "
                    suffix = suffix.strip()
                    replacement = random.choice(json_data[key]['array'])
                    new_line = line.replace(suffix, replacement, 1)
                    processed_lines.append(new_line)
                    break
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    return processed_lines

def calculate_cosine_similarity(data_demographic, data_generative):
    result = {}
    for key in data_demographic:
        if key in data_generative:
            result[key] = {}
            total_similarity = 0
            count = 0
            for sub_key in data_demographic[key]:
                if sub_key in data_generative[key]:
                    vec1 = list(data_demographic[key][sub_key].values())
                    vec2 = list(data_generative[key][sub_key].values())
                    similarity = 1 - cosine(vec1, vec2)
                    result[key][sub_key] = similarity
                    total_similarity += similarity
                    count += 1
            result[key]['total'] = result[key]['gender'] * 0.35 + result[key]['race'] * 0.35 + result[key]['age'] * 0.3
    return result

def merge_json_files(data1, data2):
    result = {}
    for key in data1:
        if key in data2:
            result[key] = {
                "cos": data1[key],
                "array": data2[key]
            }
    return result

def discriminator(input_truth, input_generative, 
                  origin_txt_path, origin_array_path, 
                  output_prompt_file_path, json_array_path, 
                  epoch):
    #set output file names and read array
    output_json = os.path.join(json_array_path, f'result_{epoch}.json')
    output_txt = output_prompt_file_path + f'prompt_epoch{epoch}.txt'
    if epoch == 1:
        input_txt = origin_txt_path
    else:
        input_txt = output_prompt_file_path + f'prompt_epoch{epoch - 1}.txt'
    if epoch == 1:
        input_array  = origin_array_path
        data_array = read_json_file(input_array)
    else:
        input_array = os.path.join(json_array_path, f"result_{epoch - 1}.json")
        last_result = read_json_file(input_array)
        data_array = get_array(last_result)

    #read the truth and generative jsons
    data_demographic_0 = read_json_file(input_truth)
    data_generative_0 = read_json_file(input_generative)
    #modify the truth and generative jsons and get difference
    data_demographic = modify_json_keys(data_demographic_0)
    data_generative = modify_json_keys(data_generative_0)
    difference = do_difference(data_demographic, data_generative) #difference < 0 means less than truth, so add protected attribute
    cosine_similarity = calculate_cosine_similarity(data_demographic, data_generative)

    #modify the array
    data_array_1 = modify_array(data_array, difference)
    data_array_2 = process_json(data_array_1)

    #merge the file
    merged_data = merge_json_files(cosine_similarity, data_array_2,)

    #process the txt
    txt_lines = read_txt_file(input_txt)
    processed_lines = process_prompts(merged_data, txt_lines)

    #output results
    os.makedirs(json_array_path, exist_ok=True)
    write_txt_file(output_txt, processed_lines)
    write_json_file(merged_data, output_json)






