import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Find all results_report.json files in the current directory and subdirectories
json_files = []
for root, dirs, files in os.walk('bloomzOnCohere'):
    for file in files:
        if file == 'results_report.json':
            json_files.append(os.path.join(root, file))

# Step 2: Extract relevant metrics from each JSON file
data = []
for file in json_files:
    print("file : ", file)
    with open(file, 'r') as f:
        json_data = json.load(f)
        #print(json_data)
        #percentage = file.split('/')[-1][-5:-4]  # assuming percentage is the last character before the extension
        for key, value in json_data.items():
            if 'percentage' in key:
                #print(f"key, value {key} {value}")
                percentage = int(value * 100)
                percentageValue = int(value * 1920)
            if 'results' in key or 'Evaluation' in key:
                metrics = {}
                metrics['test_data'] = key
                for metric_key, metric_value in value.items():
                    if 'accuracy' in metric_key:
                        metrics['Accuracy'] = metric_value
                    elif 'precision' in metric_key:
                        metrics['Precision'] = metric_value
                    elif 'recall' in metric_key:
                        metrics['Recall'] = metric_value
                    elif 'f1' in metric_key:
                        metrics['F1 Score'] = metric_value
                    elif 'sensitivity' in metric_key:
                        metrics['Sensitivity'] = metric_value
                    elif 'specificity' in metric_key:
                        metrics['Specificity'] = metric_value
                metrics['percentage'] = percentage
                metrics['percentageValue'] = percentageValue
            
                data.append(metrics)

# Step 3: Store the data in a DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by='percentage')
df['percentage'] = df['percentage'].astype(str) + '% (' + df['percentageValue'].astype(str) + ')'
df.to_csv('./dataframe.csv', index=False)


# print(df.head())
# metrics = ['accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']

# # Create and save plots
# for test_data in df['test_data']:
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, x='percentage', y=df.columns[2:], hue=test_data, marker='o')
#     plt.title(f'Percentage vs {test_data.capitalize()} for each Test Data Type')
#     plt.xlabel('Percentage')
#     plt.ylabel(test_data.capitalize())
#     plt.legend(title='Test Data Type')
#     plt.grid(True)
#     plt.savefig(f'percentage_vs_{test_data}.png')
#     plt.close()

# print("Plots have been created and saved successfully.")

# # Plotting and saving the plot with test_data names
# fig, ax = plt.subplots(figsize=(15, 10))

# for column in df.columns[2:]:
#     for i, test_data in enumerate(df['test_data']):
#         ax.plot(df['percentage'][i], df[column][i], label=f'{test_data} - {column}', marker='o')

# ax.set_xlabel('Percentage')
# ax.set_ylabel('Values')
# ax.set_ylim(0, 1)
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
# ax.set_title('Percentage vs Metrics for Different Test Data')

# # Save the plot
# fig.savefig('roberta/percentage_vs_metrics.png')

# plt.show()





# Iterate over each unique 'test_data' value
for test in df['test_data'].unique():
    # Filter data for the specific test_data
    filtered_df = df[df['test_data'] == test]
    
    # Drop the 'test_data' column for plotting
    plot_df = filtered_df.drop(columns=['test_data'])
    plot_df = plot_df.drop(columns=['percentageValue'])
    
    # Melt the DataFrame to have 'percentage' vs 'metric' for plotting
    plot_df1 = plot_df.melt(id_vars='percentage', var_name='Metric', value_name='Value')
    #plot_df2 = plot_df.melt(id_vars='percentageValue', var_name='Metric', value_name='Value')
    
    # Create the plot
    plt.figure(figsize=(8, 5))

    # Create the first plot with the left y-axis (Metric Value)
    ax1 = sns.lineplot(x='percentage', y='Value', hue='Metric', data=plot_df1, marker='o')

    # Customize the first axis (left y-axis)
    ax1.set_xlabel("Bloomz Detector Models Trained on Increasing Training Dataset Sizes")
    ax1.set_ylabel("Metric Value")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(plot_df1['percentage'].unique())
    ax1.set_xticklabels(plot_df1['percentage'].unique(), rotation=90, ha='center')
    ax1.set_title('Bloomz Detector subset-trained on M4 Cohere dataset')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # Create the second y-axis (right y-axis) for 'percentageValue'
    #ax2 = ax1.twinx()

    # Assuming 'percentageValue' is another column in plot_df
    # Plot on the second y-axis
    #sns.lineplot(x='percentageValue', y='Metric', data=plot_df2, ax=ax2, color='r', marker='o')

    #ax2.set_ylabel("Metric Value")
    #ax2.set_ylim(0, 1)
    #ax2.set_xticks(plot_df2['percentageValue'].unique())
    #ax2.set_xticklabels(plot_df2['percentageValue'].unique(), rotation=90, ha='center')

    # Adjust the layout to make sure everything fits well
    plt.tight_layout()

    # Save the plot with the test_data name
    sanitized_name = test.replace(' ', '_').replace('/', '_')
    plt.savefig(f'bloomzOnCohere/{sanitized_name}.png')
    plt.close()


# Provide file names
[sanitized_name for sanitized_name in df['test_data'].unique()]