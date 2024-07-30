import json

# Define the path to the JSON file
json_file_path = '/data/hanjun/VersusDebias/GAM_result/record/discriminator/more_result/sdv1_50/best_result.json'

# Initialize variables to store sums and counts
total_gender = 0
total_race = 0
total_age = 0
count = 0

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Iterate over each entry in the JSON data
for entry in data.values():
    cos_values = entry['cos']
    total_gender += cos_values['gender']
    total_race += cos_values['race']
    total_age += cos_values['age']
    count += 1

# Calculate averages
avg_gender = total_gender / count
avg_race = total_race / count
avg_age = total_age / count

# Calculate the weighted sum
weighted_sum = 0.35 * avg_gender + 0.35 * avg_race + 0.3 * avg_age

# Print the results
print(f"Average Gender: {avg_gender}")
print(f"Average Race: {avg_race}")
print(f"Average Age: {avg_age}")
print(f"Weighted Sum: {weighted_sum}")