import json
import os

def best_result(results_path):
    results = os.listdir(results_path)
    best_result_file = os.path.join(results_path, "best_result.json")
    
    if "best_result.json" in results:
        os.remove(best_result_file)
        results.remove("best_result.json")
    
    best_results = {}
    with open(os.path.join(results_path, results[0]), "r") as file:
        temp_result = json.load(file)

    for prompt in temp_result:
        best_results[prompt] = {"cos": {"gender": 0.0, "race": 0.0, "age": 0.0, "total": 0.0}, "array": []}
        for result in results:
            with open(os.path.join(results_path, result), "r") as file:
                result_data = json.load(file)
            prompt_data = result_data[prompt]
            if best_results[prompt]["cos"]["total"] < prompt_data["cos"]["total"]:
                best_results[prompt] = prompt_data

    with open(best_result_file, "w") as file:
        json.dump(best_results, file, indent=4)

def calculate_totals(best_result_file):
    with open(best_result_file, 'r') as file:
        json_data = json.load(file)

    totals = [entry["cos"]["total"] for entry in json_data.values()]

    average_total = sum(totals) / len(totals)
    max_total = max(totals)
    min_total = min(totals)

    print(f"Average Total: {average_total}")
    print(f"Max Total: {max_total}")
    print(f"Min Total: {min_total}")

def main(results_path):
    best_result(results_path)
    best_result_file = os.path.join(results_path, "best_result.json")
    calculate_totals(best_result_file)

if __name__ == "__main__":
    model = "lcm" # change to your generator model used in GAM
    results_path = f"./GAM_result/record/discriminator/{modle}" 
    main(results_path)
