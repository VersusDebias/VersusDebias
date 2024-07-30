import random


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def write_txt_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def enhance_prompts(lines):
    main_body = [
        "exposed face", "looking at the camera", "face in frame", "face clearly shown", "visible face", "direct eye contact", "full face view", "detailed face", "clear face", "emphasize facial feature"
    ]

    single = [
         "alone", "1person", "unaccompanied", "individually", "solo"
    ]
    quality = [
         "hdr", "high resolution", "RAW photo", "8K", "realistic skin texture", "ultra quality", "sharp focus", "masterpiece", "highly detailed", "tack sharp", "best quality", "ultra-fine", "DSLR", "4k", "UHD", "Ultra Detailed", "high dynamic range"
    ]

    style = [
        "natural lighting", "vivid colors",  "rim lighting", "photorealistic", "balanced composition", "cinematic look", "studio lighting", "Portrait", "Photojournalism Photography", "Photorealistic", "Authentic", "Filmic", "lifelike", "dof", "depth of focus"
    ]

    enhanced_lines = []
    for line in lines:
        if line.strip().startswith("--prompt"):
            enhanced_line = line.strip().strip('"')
            selected_enhancements = random.sample(main_body, 1) + random.sample(quality, 2) + random.sample(style, 2) + random.sample(single, 1)
            for enhancement in selected_enhancements:
                enhanced_line += f", {enhancement}"
            enhanced_line = f'{enhanced_line}"\n'
            enhanced_lines.append(enhanced_line)
    return enhanced_lines


input_file = './debias/data/proto_prompt.txt'
output_file = './debias/data/prompt.txt'
lines = read_txt_file(input_file)
enhanced_lines = enhance_prompts(lines)
write_txt_file(output_file, enhanced_lines)
