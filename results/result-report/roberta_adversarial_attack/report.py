import pandas as pd
import glob
import os

folder_path = './'
        # Start the HTML
html_content = """
<html>
<head>
    <title>Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Report</h1>
    <table>
        <tr>
            <th>Key</th>
            <th>Value</th>
        </tr>"""


# Add the images to the HTML
html_content += """
    </table>
    <h1>Plots/Confusion Matrix</h1>"""

# Loop through the folder and add PNG files
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png'):
        file_path = filename
        html_content += f"""
        <h2>{filename}</h2>
        <img src="{file_path}" alt="{filename}">"""

html_content += """
<h1>Original text Vs Perturbed Text</h1>"""

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        html_content += f"""
        <h2>{filename}</h2>
        <table>
            <tr>"""
        # Add the table headers
        for col in df.columns:
            html_content += f"<th>{col}</th>"
        html_content += "</tr>"
        
        # Add the table rows
        for index, row in df.iterrows():
            html_content += "<tr>"
            for col in df.columns:
                html_content += f"<td>{row[col]}</td>"
            html_content += "</tr>"
        html_content += "</table>"

# Loop through the folder to find and add TXT files
html_content += """
    <h1>Attack Summary</h1>"""

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            html_content += f"""
            <h2>{filename}</h2>
            <table>"""
            for line in lines:
                html_content += f"""
                <tr>
                    <td>{line.strip()}</td>
                </tr>"""
            html_content += "</table>"
        
# End the HTML
html_content += """
</body>
</html>"""

# Write the HTML to a file
with open(f'{folder_path}/results.html', 'w') as file:
    file.write(html_content)

print(f"Report generated: {folder_path}/results.html")
