import os
import shutil

def build_directory(prompt_file_path, image_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(prompt_file_path, 'r') as file:
        prompts = file.readlines()

    for prompt in prompts:
        prompt_content = prompt.strip()
        if '--prompt' in prompt_content:
            # Extract the entire content inside the quotes
            start_quote = prompt_content.find('"') + 1
            end_quote = prompt_content.find('"', start_quote)
            full_prompt = prompt_content[start_quote:end_quote]

            # Extract the part of the prompt before the first comma
            first_comma = full_prompt.find(',')
            prompt_key = full_prompt[:first_comma] if first_comma != -1 else full_prompt

            # Normalize the prompt key to avoid creating unnecessary folders
            prompt_key_normalized = normalize_prompt_key(prompt_key)

            # Create directory for this prompt
            folder_name = prompt_key_normalized.strip().replace(" ", "_")  # Use the part before the first comma as folder name
            folder_path = os.path.join(output_directory, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Copy relevant files to the corresponding folder
            for filename in os.listdir(image_directory):
                if prompt_key in filename:
                    src_path = os.path.join(image_directory, filename)
                    dest_path = os.path.join(folder_path, filename)
                    shutil.move(src_path, dest_path)

def normalize_prompt_key(prompt_key):
    # Remove extra descriptors or modifiers and normalize the prompt key
    keywords_to_remove = ["Indian ", "Black ", "White ", "East Asian ", "Southeast Asian ", "Latino_Hispanic ", "Middle Eastern ","young ","elderly ","middle-aged "]
    normalized_key = prompt_key
    for keyword in keywords_to_remove:
        normalized_key = normalized_key.replace(keyword, "")
    if "woman" in normalized_key:
        normalized_key = normalized_key.replace("woman", "person")
    elif "man" in normalized_key:
        normalized_key = normalized_key.replace("man", "person")
    return normalized_key

# Example usage
if __name__ == "__main__":
    prompt_file_path = '/data/hanjun/VersusDebiaser_v1/generator/prompt/prompts_final.txt'
    image_directory = '/data/hanjun/comfy/output'
    output_directory = '/data/hanjun/debiaser/tie0/cascade'
    build_directory(prompt_file_path, image_directory, output_directory)
