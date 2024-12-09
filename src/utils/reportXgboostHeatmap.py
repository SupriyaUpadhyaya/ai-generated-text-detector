import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read CSV file into a DataFrame

data = pd.read_csv('model_metrics_results.csv', index_col=0)

x_labels = data.columns[~data.columns.str.contains('^Unnamed')]

# Set the first row as column headers and remove it from the data
data.columns = data.iloc[0]
data = data.drop(data.index[0])

# Replace 'na' values with NaN
data.replace('na', np.nan, inplace=True)

# Round the values to two decimal places
data = data.astype(float).round(2)

# Define function to create and save heatmap
def create_heatmap(df, metric_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='Blues', xticklabels=x_labels, mask=df.isnull())
    plt.title(f'Heatmap for {metric_name}')
    plt.xlabel("Evaluation dataset")
    plt.ylabel("Training dataset")
    plt.tight_layout()
    plt.savefig(f'heatmap_{metric_name}.png')
    plt.show()

# Split the dataframe into separate parts for each metric (assuming columns represent precision, recall, accuracy, f1)
precision_data = data[['precision']]
recall_data = data[['recall']]
#accuracy_data = data[['accuracy']]
f1_score_data = data[['f1']]

# Create heatmaps for each metric
create_heatmap(precision_data, "Precision")
create_heatmap(recall_data, "Recall")
#create_heatmap(accuracy_data, "Accuracy")
create_heatmap(f1_score_data, "F1 Score")

