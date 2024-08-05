#!/usr/bin/bash

# Check if modelType and trainData arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <modelType> <trainData>"
    exit 1
fi

# Assign arguments to variables
modelType=$1
trainData=$2

# Define the percentages
percentages=(1 2 4 8 10 15 20 25 30 35 40)
#percentages=(45 50 60 70 80 90 100)

# Loop through each percentage
for percentage in "${percentages[@]}"; do
    # Calculate the value for --percentage
    percentage_value=$(awk "BEGIN {print $percentage / 100}")
    
    # Define the log folder name
    log_folder_name="${modelType}_trainOnSubset_eval_$percentage"
    
    # Define the log file name
    log_file="./results/logs/${modelType}_trainOnSubset_eval_$percentage.log"
    
    # Define the title
    title="${modelType} Finetune on ${trainData} with ${percentage}% Data and cross-model Evaluation"
    
    # Print the current percentage and calculated value
    echo "Processing percentage: $percentage"
    echo "Calculated percentage value: $percentage_value"
    
    # Print the log folder and file names
    echo "Log folder name: $log_folder_name"
    echo "Log file: $log_file"
    
    # Print the title
    echo "Title: $title"
    
    # Run the command and print the command being executed
    echo "Executing command: python3 main.py --modelType $modelType --task trainOnSubset --percentage \"$percentage_value\" --trainData $trainData --dataType abstract --newLine without --log_folder_name \"$log_folder_name\" --title \"$title\""
    python3 main.py \
        --modelType "$modelType" \
        --task "trainOnSubset" \
        --percentage "$percentage_value" \
        --trainData "$trainData" \
        --dataType "abstract" \
        --newLine "without" \
        --log_folder_name "$log_folder_name" \
        --title "$title"
    
    # Print the PID of the background process
    echo "Started background process with PID $!"
done
