import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
data = pd.read_csv('xgboost.csv')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Hide axes
ax.axis('off')

# Create a table and add it to the axes
table = ax.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')

# Scale the table
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)

# Save the figure as an image
plt.savefig('csv_as_image.png', bbox_inches='tight', dpi=1000)
plt.close()
