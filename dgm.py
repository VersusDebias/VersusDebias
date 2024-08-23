import debias
import os
import shutil

def find_overlap(prompt_content, str2):

    return overlap
def build_directory(prompt_file_path, ground_truth, image_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(prompt_file_path, 'r') as file:
        prompts = file.readlines()

    for prompt in prompts:
        prompt_content = prompt.strip()
        if '--prompt' in prompt_content:
            start_quote = prompt_content.find('"') + 1
            end_quote = prompt_content.find('"', start_quote)
            full_prompt = prompt_content[start_quote:end_quote]
            folder_path = os.path.join(output_directory, full_prompt)    
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Copy relevant files to the corresponding folder
            for filename in os.listdir(image_directory):
                # check overlap
                max_len = 0
                overlap = ""
                for i in range(len(prompt_content)):
                    for j in range(len(prompt_content)-i+1):
                        if prompt_content[i:i+j] in filename and j > max_len:
                            max_len = j
                            overlap = prompt_content[i:i+j]
                if overlap != "":
                    src_path = os.path.join(image_directory, filename)
                    dest_path = os.path.join(folder_path, filename)
                    shutil.move(src_path, dest_path)
                    break
def main():
    # choose your prompt set(.txt file)
    # if you want to use eval, select from the foolowing:
    # precise_debias : "./evaluate/eval_pd.txt"
    # zeroshot : "./evaluate/eval_zeroshot.txt"
    # fewshot : "./evaluate/eval_fewshot.txt"
    original_prompt_path = "your/prompt_set.txt"
    # choose your generator_model
    generator_model = "lcm"
    # choose your ground truth path
    # if you want to use eval, select from the foolowing:
    # precise_debias : "./evaluate/eval_pd_gt.json"
    # zeroshot : "./evaluate/eval_zeroshot_gt.json"
    # fewshot : "./evaluate/eval_fewshot_gt.json"
    ground_truth = "your/gt.json"

    debiased_prompt_path = "./prompt/debiased_prompt.txt"
    best_result_path = f"./GAM_result/record/discriminator/{generator_model}/best_result.json"
    element_path = "./debias/data/element.txt"
    slm_model_path = "./model/qwen1_5b-executor"
    print("Execute: Debiasing prompt")
    debias.execute(original_prompt_path,
                debiased_prompt_path,
                best_result_path,
                element_path,
                slm_model_path)
    print("Debias finished, start generating images")
    # generator
    json_template_path = f"./workflow/{generator_model}.json" # make sure you provide correct form of json
    server_address = "127.0.0.1:8197"
    debias.generate_images_dgm(original_prompt_path,
                                debiased_prompt_path,
                                json_template_path,
                                server_address)
    # This path is the output path of your comfy2
    image_directory = "your/comfy2/output/path"
    output_directory = f"./Debiased_Image/image/{generator_model}"
    build_directory(original_prompt_path,
                    ground_truth,
                    image_directory,
                    output_directory)

if __name__ == "__main__":
    main()
