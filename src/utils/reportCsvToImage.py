import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to create and save an image from a CSV file
def csv_to_image(csv_file_path, output_image_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create a table from the DataFrame
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Adjust the font size and scale for better display
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    # Save the table as an image
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    plt.close()

# Walk through all folders and subfolders to find CSV files
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            csv_file_path = os.path.join(root, file)
            output_image_path = os.path.splitext(csv_file_path)[0] + '.png'
            csv_to_image(csv_file_path, output_image_path)
            print(f"Saved image for {csv_file_path} as {output_image_path}")
