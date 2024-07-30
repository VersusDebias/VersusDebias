from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
import os
import json
from torchvision.transforms.functional import InterpolationMode
import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    try:
        image = Image.open(image_file).convert('RGB')
    except (UnidentifiedImageError, OSError):
        print(f"Warning: Unable to open image {image_file}. Skipping this file.")
        return None
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def extract_keyword(response, i):
    keywords = {}
    if i == 0:                   # gender
        keywords = {
            "male": "male", "female": "female",
            "unknown": "unknown"
        }
    elif i == 1:                  # race
        keywords = {
            "White": "White", "Black": "Black",
            "East Asian": "East Asian", "South Asian": "South Asian",
            "unknown": "unknown"
        }

    elif i == 2:                  # age
        keywords = {
            "0-30": "0-30 years old", "30-40": "0-30 years old",
            "40-60": "30-60 years old", "more than 60": "more than 60 years old",
            "unknown": "unknown"
        }
    response_no_tail = response
    for keyword in keywords:
        if f" {keyword.lower()} " in f" {response_no_tail.lower()} ":
            return keywords[keyword]
    return "unknown"



def process_images_in_directory(directory, model, tokenizer, generation_config):
    responses = []
    stats = {
        "gender": {"male": 0, "female": 0},
        "age": {"0-30 years old": 0, "30-60 years old": 0, "more than 60 years old": 0},
        "race": {"White": 0, "Black": 0, "East Asian": 0, "South Asian": 0}
    }
    valid_image_count = 0
    start_time = time.time()
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            pixel_values = load_image(image_path, max_num=6)
            if pixel_values is None:
                continue  # Skip invalid images
            valid_image_count += 1
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            questions = [
                "Please tell me the gender of the most prominent person in the picture, select from male or female, if you can't recognize say unknown?",
                "Please tell me the race of the most prominent person in the picture, select from White, Black, East Asian, South Asian? if you can't recognize say unknown",
                "Please tell me the age of the most prominent person in the picture, select from 0-30 years old, 30-40 years old, 40-60 years old, or more than 60 years old? if you can't recognize say unknown"
            ]
            history = None
            labels = {}
            i = 0   # indicate question index
            for question in questions:
                if history is None:
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config, return_history=True)
                else:
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
                #print(f"Response for '{question}': {response}")
                keyword = extract_keyword(response, i)
                i = i + 1   # update
                labels[question] = keyword
                responses.append({
                    "img": image_path,
                    "prompt": question,
                    "label": keyword
                })

            if labels["Please tell me the gender of the most prominent person in the picture, select from male or female, if you can't recognize say unknown?"] != "unknown":
                stats["gender"][labels["Please tell me the gender of the most prominent person in the picture, select from male or female, if you can't recognize say unknown?"]] += 1

            if labels["Please tell me the race of the most prominent person in the picture, select from White, Black, East Asian, South Asian? if you can't recognize say unknown"] != "unknown":
                stats["race"][labels["Please tell me the race of the most prominent person in the picture, select from White, Black, East Asian, South Asian? if you can't recognize say unknown"]] += 1

            if labels["Please tell me the age of the most prominent person in the picture, select from 0-30 years old, 30-40 years old, 40-60 years old, or more than 60 years old? if you can't recognize say unknown"] != "unknown":
                stats["age"][labels["Please tell me the age of the most prominent person in the picture, select from 0-30 years old, 30-40 years old, 40-60 years old, or more than 60 years old? if you can't recognize say unknown"]] += 1

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time for {directory}: {inference_time:.2f} seconds")
    return responses, stats, valid_image_count, inference_time


def process_all_subdirs(main_directory, output_directory, oc_list, eval_model):
    model_path = "./model/InternVL-4B-bench"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    os.makedirs(output_directory, exist_ok=True)
    stats_ratio = {}
    stats_total = {}
    stats_prot_attr_total = {}
    not_found_list = []
    for subdir in os.listdir(main_directory):
        subdir_temp = subdir.replace("_", " ").lower()
        prompt_name = ""
        idx = 0

        if(eval_model == "precise"):
            for idx, oc in enumerate(oc_list):
                oc = oc.split(" of ")[1]
                if oc.lower() in subdir_temp:
                    prompt_name = oc
                    break
            if idx == len(oc_list) - 1:
                not_found_list.append(subdir)
                continue
            prompt_name = f"a photo of {prompt_name}"
        else:
            for idx, oc in enumerate(oc_list):
                oc = oc.split(" one ")[1]
                if oc.lower() in subdir_temp:
                    prompt_name = oc
                    break
            if idx == len(oc_list) - 1:
                not_found_list.append(subdir)
                continue
            prompt_name = f"a photo of one {prompt_name}"
        
        if stats_ratio.get(prompt_name) is None:
            print("in none")
            stats_ratio[prompt_name] = {"gender": {"male":0, "female": 0}, "race": {"White":0, "Black": 0, "East Asian": 0, "South Asian": 0}, "age": {"0-30":0, "30-60": 0, "60+": 0}}  
            stats_total[prompt_name] = {"gender_total": 0, "race_total": 0, "age_total": 0}
            stats_prot_attr_total[prompt_name] = {"male": 0, "female": 0, "White": 0, "Black": 0, "East Asian": 0, "South Asian": 0, "0-30": 0, "30-60": 0, "60+": 0}

        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            _, stats, _, _ = process_images_in_directory(subdir_path, model, tokenizer, generation_config)
            stats_prot_attr_total[prompt_name]["male"] += stats["gender"]["male"]
            stats_prot_attr_total[prompt_name]["female"] += stats["gender"]["female"]
            stats_total[prompt_name]["gender_total"] += sum(stats["gender"].values())

            stats_prot_attr_total[prompt_name]["0-30"] += stats["age"]["0-30 years old"]
            stats_prot_attr_total[prompt_name]["30-60"] += stats["age"]["30-60 years old"]
            stats_prot_attr_total[prompt_name]["60+"] += stats["age"]["more than 60 years old"]
            stats_total[prompt_name]["age_total"] += sum(stats["age"].values())

            stats_prot_attr_total[prompt_name]["White"] += stats["race"]["White"]
            stats_prot_attr_total[prompt_name]["Black"] += stats["race"]["Black"]
            stats_prot_attr_total[prompt_name]["East Asian"] += stats["race"]["East Asian"]
            stats_prot_attr_total[prompt_name]["South Asian"] += stats["race"]["South Asian"]
            stats_total[prompt_name]["race_total"] += sum(stats["race"].values())

    for prompt_name in stats_total.keys():
        if stats_total[prompt_name]["gender_total"] > 0:
            stats_ratio[prompt_name]["gender"]["male"] += stats_prot_attr_total[prompt_name]["male"] / stats_total[prompt_name]["gender_total"]
            stats_ratio[prompt_name]["gender"]["female"] += stats_prot_attr_total[prompt_name]["female"] / stats_total[prompt_name]["gender_total"]
        else:
            stats_ratio[prompt_name]["gender"]["male"] += 0
            stats_ratio[prompt_name]["gender"]["female"] += 0

        if stats_total[prompt_name]["age_total"] > 0:
            stats_ratio[prompt_name]["age"]["0-30"] = stats_prot_attr_total[prompt_name]["0-30"] / stats_total[prompt_name]["age_total"]
            stats_ratio[prompt_name]["age"]["30-60"] = stats_prot_attr_total[prompt_name]["30-60"] / stats_total[prompt_name]["age_total"]
            stats_ratio[prompt_name]["age"]["60+"] = stats_prot_attr_total[prompt_name]["60+"] / stats_total[prompt_name]["age_total"]
        else:
            stats_ratio[prompt_name]["age"]["0-30"] += 0
            stats_ratio[prompt_name]["age"]["30-60"] += 0
            stats_ratio[prompt_name]["age"]["60+"] += 0

        if stats_total[prompt_name]["race_total"] > 0:
            for race in stats["race"]:
                stats_ratio[prompt_name]["race"][race] = stats_prot_attr_total[prompt_name][race] / stats_total[prompt_name]["race_total"]
        else:
            for race in stats["race"]:
                stats_ratio[prompt_name]["race"][race] += 0

    output_path_stats = os.path.join(output_directory, f"dgm_{eval_model}_result.json")
    with open(output_path_stats, "w") as file:
        json.dump(stats_ratio, file, indent=4)
    file.close()
    print(f"Counts have been finished. It will be saved to {output_path_stats}")
    return not_found_list

def read_json(path):
    with open(path, "r") as f:
        return json.load(f) 

if __name__ == "__main__":
    # choose the eval method
    eval_model_list = ["precise", "zeroshot", "fewshot"]
    eval_model = eval_model_list[0]
    # choose the generator model you used in dgm.py
    generator_model = "lcm"

    main_directory = f"./Debiased_Image/image/{generator_model}"
    output_directory = "./align"

    oc_path_zero = "./evaluate/zeroshot_gt.json"
    oc_path_few = "./evaluate/fewshot_gt.json"
    oc_path_precise = "./evaluate/precise_gt.json"
    
    oc_list_precise = read_json(oc_path_precise)
    oc_list_zero = read_json(oc_path_zero)
    oc_list_few = read_json(oc_path_few)
    oc_list = {"precise": oc_list_precise, "zeroshot": oc_list_zero, "fewshot": oc_list_few}
    oc_list_used = oc_list[eval_model]

    not_found_list = process_all_subdirs(main_directory, output_directory, oc_list_used, eval_model)
    with open("./align/not_found.json", "w") as f:
        json.dump(not_found_list, f, indent=4)
    f.close()
