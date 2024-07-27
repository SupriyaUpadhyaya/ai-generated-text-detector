import pandas as pd

# Read the CSV file
file_path = '/Users/supriyaupadhyaya/Desktop/attack_results_ai_pwws.csv'
df = pd.read_csv(file_path)

# Check if required columns are present
if 'original_text' in df.columns and 'perturbed_text' in df.columns:
    # Generate HTML code
    html_content = """
    <html>
    <head>
        <title>Text Comparison</title>
        <style>
            table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h2>Text Comparison</h2>
        <table>
            <tr>
                <th>Original Text</th>
                <th>Perturbed Text</th>
            </tr>
    """

    # Add table rows
    for _, row in df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['original_text']}</td>
                <td>{row['perturbed_text']}</td>
            </tr>
        """

    # Close the table and HTML tags
    html_content += """
        </table>
    </body>
    </html>
    """

    # Save HTML to file
    with open('text_comparison.html', 'w') as file:
        file.write(html_content)
    
    print("HTML file has been created successfully!")
else:
    print("CSV file does not contain required columns 'original_text' and 'perturbed_text'")
