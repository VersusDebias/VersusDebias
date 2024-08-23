##call module test
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import re

def chat1_5(model, tokenizer, ques, history=[], temperature = 0.7):
    if history == []:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ques}
        ]
    else:
        messages = history
        messages.append({"role": "user", "content": ques})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature = temperature
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    messages.append({
        "role": "system", "content": response
    })
    
    history = messages

    return response, history

def extract_descriptors(prompts, occupation):
    descriptors = set()  
    man_woman_pattern = re.compile(r"(.+?)\s(man|woman)", re.IGNORECASE)

    if occupation.endswith("person"):
        for prompt in prompts:
            match = man_woman_pattern.search(prompt)
            if match:
                descriptors.add(match.group(2))
            else:
                occupation_pattern = re.compile(r"(.+?)\s" + re.escape(occupation), re.IGNORECASE)
                match = occupation_pattern.search(prompt)
                if match:
                    descriptors.add(match.group(1).strip())
    else:
        occupation_pattern = re.compile(r"(.+?)\s" + re.escape(occupation), re.IGNORECASE)
        for prompt in prompts:
            match = occupation_pattern.search(prompt)
            if match:
                descriptors.add(match.group(1).strip())

    return list(descriptors)

def read_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("filenot found")
    except json.JSONDecodeError:
        print("json decode error")  
    
def find_occupation_json(occupation, data):
    # lowercase
    lower_case_data = {key.lower(): value for key, value in data.items()}
    occupation = occupation.lower()  
    if occupation in lower_case_data:
        random_element = random.choice(lower_case_data[occupation]['array'])
        list_prompt = [random_element]
        print(f"prompt is {list_prompt}")
        print(occupation)
        description = extract_descriptors(list_prompt,occupation)
        return description
    else:
        return None

def read_occupations(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read all lines in the file
        data = file.read()
        # Split the data into a list based on commas
        occupation_list = data.split(',')
        # Strip any leading or trailing whitespace from each item
        occupation_list = [occupation.strip() for occupation in occupation_list]
    return occupation_list

def is_in_response(key, input_string):
    return key in input_string.lower()



# return 1  continue to next prompt
# return 2  there is error in the function
# return occupation answer to q3  
def minicpm_response(prompt, model, tokenizer):
    # Seed for reproducibility
    CONTINUE_SIGNAL = 1
    SUCCESS = 0
    ERROR = 2
    torch.manual_seed(0)

    q1 = f"Get ONLY the occupation from the prompt: {prompt}. Your answer format should be 'Occupation: '"
    q2 = f"Whether there is already a EXPLICIT protected attribute describing 'race or gender or age' in the prompt given by me in previous question, yes or no? including but not limited to: male, female, man, woman, boy, girl, White, Black, East-Asian, South-Asian, Caucasian, Latino, Hispanic, young, old, middle-aged, and so on."
    q3 = f"For the occupation answer in your response to the first question, select the element that is synonym of the answer from the list that follows. ANSWER ONLY WITH THE KEYWORD: chief executive,manager,marketing manager,human resource worker,accountant,production manager,transportation manager,farmer,construction manager,education administrator,business agent,purchasing agent,insurance worker,financial analyst,courier,public relations specialists,computer programmer,computer scientist,it analyst,mathematician,architect,civil engineer,electrical engineer,industrial engineer,mechanical engineer,drafter,surveyor,biological scientist,agricultural scientist,environmental scientist,chemist,astronomer,physicist,geoscientist,sociologist,psychologist,technician,mental counselor,social worker,clergy,lawyer,judge,legal assistant,school teacher,tutor,librarian,artist,designer,actor,director,athlete,coach,dancer,musician,broadcast announcer,journalist,writer,interpreter,photographer,doctor,nutritionist,pharmacist,therapist,veterinarian,nurse,paramedic,health aide,correctional officer,firefighter,detective,police officer,security guard,cook,food server,cleaner,animal trainer,concierge,mortician,barber,travel guide,exercise trainer,cosmetologist,cashier,salesperson,vendor,telephone operator,office clerk,secretary,construction worker,painter,metal worker,miner,repairer,factory worker,food worker,transportation attendant,transportation operator,driver,sailor,refuse collector,laborer,politician,tailor,woodworker"
    print(prompt)
    # Define the questions
    questions = [
        q1,
        q2,
        q3
    ]

    # Initialize the chat history
    # history = None
    history = []

    # Iteratively ask each question
    q1_response = " "
    for i, question in enumerate(questions):
        response, history = chat1_5(
            model,
            tokenizer,
            question,
            history=history,  # Passing the current history
            # top_p=0.8,
            # repetition_penalty=1.02
        )
        if(i==0):
            print(f"q1 response: {response}")
            q1_response = response
            keyword = "none"
            if is_in_response(keyword, response):
                print("q1-skip")
                return CONTINUE_SIGNAL
            
        elif(i==1):
            print(f"q2 response: {response}")
            keyword = "yes"
            if is_in_response(keyword, response):
                print("q2-skip")
                return CONTINUE_SIGNAL
            
        else:
            print(f"q3 response: {response}")
            occupation_original = q1_response
            occupation = response
            parts_r3_occupation = occupation
            if "Occupation" in occupation_original and "Characteristic" not in occupation_original:
                parts_r1_occupation = occupation_original.split("Occupation: ")
                if len(parts_r1_occupation) > 1:
                    return [parts_r3_occupation, parts_r1_occupation[1].strip()]            
                else:
                    return ERROR
            elif "Occupation/Characteristic" in occupation_original:
                parts_r1_occupation = occupation_original.split("Occupation/Characteristic: ")
                if len(parts_r1_occupation) > 1:
                    return [parts_r3_occupation, parts_r1_occupation[1].strip()]            
                else:
                    return ERROR
            else:
                parts_r1_occupation = occupation_original
                return [parts_r3_occupation, parts_r1_occupation.strip()]    

def execute(prompt_path,
            new_prompt_path,
            file_path,
            element_path,
            model_path):
    occupations_list = read_occupations(element_path)
    data = read_data(file_path)

    if data:
        print("load prompt")
    else: 
        print("can't load json")

    quote_pattern = re.compile(r'"(.+?)"')
    with open(prompt_path, 'r', encoding='utf-8') as file_read, \
     open(new_prompt_path, 'w', encoding='utf-8') as file_write:
        torch.manual_seed(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)    
        for line in file_read:
            CONTINUE_SIGNAL = 0
            ERROR_SIGNAL = 0
            match = quote_pattern.search(line.strip())
            if match:
                prompt = match.group(1)
                # print(prompt)
                s = minicpm_response(prompt, model, tokenizer)

                if(s == 1):
                    print("continue next prompt and save orignial prompt to new_prompt.txt")  ## continue next prompt
                    CONTINUE_SIGNAL = 1
                elif(s == 2):
                    print("error in q3 and save orignial prompt to new_prompt.txt  it is because we split occupation: in the response the error might probably becasue the response format have been changed")
                    CONTINUE_SIGNAL = 1 
                else:
                    occupation_q3 = s[0]
                    original_occupation = s[1]
                    if(occupation_q3.lower() in occupations_list):
                        attribute = find_occupation_json(occupation_q3, data)
                        if(attribute == []):
                            print("r3 in list but best result tell us no need to changed ")
                            CONTINUE_SIGNAL=1
                        elif attribute == None:
                            print("no occupation matched between r3 and best_result")
                            CONTINUE_SIGNAL=1
                        else:
                            modify_string = attribute[0]
                            gender_list = ['man', 'woman']
                            # print(modify_string)
                            change_occupation = ""
                            if (modify_string in gender_list):
                                if("person" in original_occupation.lower()):
                                    #successfully changed eg. brave person -> brave person
                                    occupation_original_lower = original_occupation.lower()
                                    change_occupation = occupation_original_lower.replace("person", modify_string)
                                else:
                                    ERROR_SIGNAL = 1
                                    print("q1 no person q3 exist man woman error")
                            else:
                                # print(modify_string)
                                # print(original_occupation)
                                change_occupation = modify_string + " " + original_occupation   ##eg. European + architect
                            #change the line if occupationq3 in occupation_list
                            line_lower = line.lower()
                            replaced_line = line_lower.replace(original_occupation.lower(), change_occupation)
                    else:   
                        print("occupation not in list but continue")  ##continue next prompt but indicate there is error   occupation not in list
                        ERROR_SIGNAL = 2                              ## error signal 2 means occupation is not in list

                if CONTINUE_SIGNAL or ERROR_SIGNAL:
                    #keep original prompt   
                    replaced_line = line
            
                file_write.write(replaced_line)

            else:
                print("No prompt found in the line of prompt.txt:", line.strip())
                continue
    
