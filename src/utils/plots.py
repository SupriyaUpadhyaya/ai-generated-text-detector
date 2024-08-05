import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Find all results_report.json files in the current directory and subdirectories
json_files = []
for root, dirs, files in os.walk('results/report/roberta'):
    for file in files:
        if file == 'results_report.json':
            json_files.append(os.path.join(root, file))

# Step 2: Extract relevant metrics from each JSON file
data = []
for file in json_files:
    
    with open(file, 'r') as f:
        json_data = json.load(f)
        #percentage = file.split('/')[-1][-5:-4]  # assuming percentage is the last character before the extension
        for key, value in json_data.items():
            if 'percentage' in key:
                percentage = int(value * 100)
            if 'results' in key or 'Evaluation' in key:
                metrics = {'test_data': None, 'percentage': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None}
                metrics['test_data'] = key
                for metric_key, metric_value in value.items():
                    if 'accuracy' in metric_key:
                        metrics['accuracy'] = metric_value
                    elif 'precision' in metric_key:
                        metrics['precision'] = metric_value
                    elif 'recall' in metric_key:
                        metrics['recall'] = metric_value
                    elif 'f1' in metric_key:
                        metrics['f1'] = metric_value
                metrics['percentage'] = percentage
            
                data.append(metrics)

# Step 3: Store the data in a DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by='percentage')
df.to_csv('dataframe.csv', index=False)


print(df.head())
metrics = ['accuracy', 'precision', 'recall', 'f1']

# Create and save plots
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='percentage', y=metric, hue='test_data', marker='o')
    plt.title(f'Percentage vs {metric.capitalize()} for each Test Data Type')
    plt.xlabel('Percentage')
    plt.ylabel(metric.capitalize())
    plt.legend(title='Test Data Type')
    plt.grid(True)
    plt.savefig(f'percentage_vs_{metric}.png')
    plt.close()

print("Plots have been created and saved successfully.")
