import json
from scipy.spatial.distance import cosine
import random
import os
import numpy as np
def calculate_cosine_similarity(data_demographic, data_generative):
    result = {}
    for key in data_demographic:
        if key in data_generative:
            total_similarity = 0
            count = 0
            check_prompt = False
            for sub_key in data_demographic[key]:
                if sub_key in data_generative[key]:
                    check = 0
                    for prot_attr in data_generative[key][sub_key]:
                        check = check + data_generative[key][sub_key][prot_attr]
                    if(check == 0):
                        check_prompt = True
                        break
            if check_prompt:
                continue
            result[key] = {}
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

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def store_json_file(file_path, data, eval_model):
    with open(os.path.join(file_path, f"result_{eval_model}.json"), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    # choose your eval method
    eval_model_list = ["precise", "zeroshot", "fewshot"]
    eval_model = eval_model_list[0]

    ground_truth_path = f"./evaluate/{eval_model}_gt.json"
    align_path = f"./align/dgm_{eval_model}_result.json"
    file_path = "./evaluate"
    ground_truth = read_json(ground_truth_path)
    align = read_json(align_path)
    result = calculate_cosine_similarity(ground_truth, align)
    store_json_file(file_path, result, eval_model)
    total_total = total_gender = total_race = total_age = 0
    len = 0
    for key in result:
        total_total += result[key]['total']
        total_gender += result[key]['gender']
        total_race += result[key]['race']
        total_age += result[key]['age']
        len += 1

    print('total:', total_total / len, 'gender:', total_gender / len, 'race:', total_race / len, 'age:', total_age / len)
        