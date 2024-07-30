import websocket 
import uuid
import json
import urllib.request
import urllib.parse
import random
import time

server_address = "127.0.0.1:8197"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  
        else:
            continue 

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            images_output = []
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def load_prompts(filename):
    prompts = []
    with open(filename, "r") as file:
        for line in file:
            if '--prompt' in line:
                start = line.find('"') + 1
                end = line.rfind('"')
                if start > 0 and end > 0:
                    prompt = line[start:end]
                    prompts.append(prompt)
    return prompts

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    file.close()
    return data

def truncate_prompt(prompt):
    parts = prompt.split(',', 3)  # 分割成最多四部分
    if len(parts) > 3:
        truncated = ','.join(parts[:3])  # 取前三部分
    else:
        truncated = prompt
    return truncated
def all_prompts_processed_dgm(prompts, names, ws, json_template_path):
    for i, prompt_text in enumerate(prompts):
        prompt = read_json_file(json_template_path)
        # If use pixart, uncomment the lines below and comment the next 3 lines
        # prompt["113"]["inputs"]["text"] = prompt_text
        # prompt["66"]["inputs"]["filename_prefix"] = names[i]
        # prompt["155"]["inputs"]["seed"] = random.randint(1000, 6000000000)
        prompt["6"]["inputs"]["text"] = prompt_text
        temp = names[i]
        prompt["9"]["inputs"]["filename_prefix"] = truncate_prompt(temp)
        prompt["3"]["inputs"]["seed"] = random.randint(1000, 6000000000)
        prompt_id = queue_prompt(prompt)['prompt_id']
        print(f"Processed prompt: {prompt_text}")
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  
            else:
                continue  
                
    return True

def generate_images_dgm(name_file_path, prompt_file_path, json_template_path, server_address="127.0.0.1:8197"):
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

    prompts = load_prompts(prompt_file_path)
    names = load_prompts(name_file_path)
    
    if len(prompts) != len(names):
        raise ValueError("The lengths of prompts and names do not match.")
    
    if all_prompts_processed_dgm(prompts, names, ws, json_template_path):
        print("All prompts have been processed successfully.")
        ws.close()
        return True