import os
import json
import csv

# Define model names to search for in JSON files
models = [
    'chatgpt_abstract_without', 'chatgpt_abstract_paraphrased_without', 'bloomz_abstract_without', 
    'bloomz_abstract_paraphrased_without', 'llama3_abstract_without', 'cohere_abstract_without', 
    'davinci_abstract_without', 'flant5_abstract_without', 'llama3_ml_without', 'bloomz_wiki_without', 
    'chatgpt_wiki_without', 'cohere_wiki_without', 'davinci_wiki_without', 
    'chatgptBloomz_abstract_without', 'bloomzWiki_abstract_without', 'bloomzWikiML_abstract_without'
]

# Metrics to extract for each model
metrics = ['precision', 'Recall', 'Accuracy', 'F1']

# Initialize a dictionary to hold the results
results = {}

# Get the list of all subdirectories in the current working directory
current_dir = os.getcwd()
for folder in os.listdir(current_dir):
    folder_path = os.path.join(current_dir, folder)
    
    if os.path.isdir(folder_path):
        # Initialize the results for this directory
        results[folder] = {f"{model}_{metric}": 'na' for model in models for metric in metrics}
        
        # Traverse through each json file in the directory
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                json_file_path = os.path.join(folder_path, file)
                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                # Loop through the keys in the JSON file
                for key, value in data.items():
                    # Check for each model and metric if the key contains both
                    for model in models:
                        for metric in metrics:
                            if model in key and metric in key:
                                # Update the result for this directory and model-metric combination
                                results[folder][f"{model}_{metric}"] = value

# Create CSV output
csv_file = 'model_metrics_results.csv'
with open(csv_file, 'w', newline='') as csvfile:
    # Define the header for CSV
    header = ['directory_name'] + [f"{model}_{metric}" for model in models for metric in metrics]
    writer = csv.DictWriter(csvfile, fieldnames=header)

    # Write header
    writer.writeheader()

    # Write the data
    for folder_name, folder_data in results.items():
        row = {'directory_name': folder_name}
        row.update(folder_data)
        writer.writerow(row)

print(f"Results saved to {csv_file}")
