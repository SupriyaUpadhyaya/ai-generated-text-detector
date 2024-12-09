import os

# Specify the directory to search for PNG files
directory = 'roberta'  # Change this to your target directory

# Get all PNG files in the specified directory
png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

# Group files based on the specified keywords
abstract_files = [f for f in png_files if 'abstract' in f]
wiki_files = [f for f in png_files if 'wiki' in f]
ml_files = [f for f in png_files if 'ml' in f]

# Start writing the HTML file with styling for 3 columns
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Roberta-SubSet-Finetune</title>
    <style>
        .image-grid {
            display: flex;
            flex-wrap: wrap;
        }
        .image-grid div {
            flex: 1 1 calc(33.333% - 10px); /* 3 columns with some margin */
            margin: 5px;
        }
        .image-grid img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>Roberta Detector trained on Bloomz generations</h1>
    
    <h2>Abstract Generations</h2>
    <div class="image-grid">
"""

# Add abstract files to the HTML with full path
for file in abstract_files:
    full_path = os.path.abspath(os.path.join(directory, file))
    html_content += f'<div><img src="{full_path}" alt="{file}"></div>\n'

html_content += "</div>\n<h2>Wiki Generations</h2>\n<div class=\"image-grid\">\n"

# Add wiki files to the HTML with full path
for file in wiki_files:
    full_path = os.path.abspath(os.path.join(directory, file))
    html_content += f'<div><img src="{full_path}" alt="{file}"></div>\n'

html_content += "</div>\n<h2>ML Generations</h2>\n<div class=\"image-grid\">\n"

# Add ml files to the HTML with full path
for file in ml_files:
    full_path = os.path.abspath(os.path.join(directory, file))
    html_content += f'<div><img src="{full_path}" alt="{file}"></div>\n'

# Close the HTML tags
html_content += """
    </div>
</body>
</html>
"""

# Write the HTML content to a file
output_file = os.path.join(directory, "roberta.html")
with open(output_file, "w") as html_file:
    html_file.write(html_content)

print(f"HTML file created: {output_file}")
