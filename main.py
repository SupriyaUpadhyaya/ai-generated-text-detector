import argparse
import torch
from run import Run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_report = {}

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some input arguments.")

    # Add arguments
    parser.add_argument("--modelType", type=str, choices=["roberta", "bloomz", "xgboost"], default="roberta", 
                        help="Type of the model: roberta, bloomz, or xgboost")
    parser.add_argument("--train", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Train the model: true or false")
    parser.add_argument("--trainData", type=str, choices=["chatgpt", "bloomz", "cohere", "flat5", "davinci"], default="chatgpt",
                        help="Training data source: chatgpt, bloomz, cohere, flat5, or davinci")
    parser.add_argument("--dataType", type=str, choices=["abstract", "wiki", "ml"], default="abstract",
                        help="Type of data: abstract, wiki, or ml")
    parser.add_argument("--newLine", type=str, choices=["with", "without"], default="without",
                        help="Include new lines in the output: with or without")
    parser.add_argument('--log_folder_name', type=str, required=True, help='Log folder name')
    parser.add_argument("--attack", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Train the model: true or false")
    parser.add_argument('--title', type=str, required=True, help='Title')

    # Parse the arguments
    args = parser.parse_args()
    model_type = args.modelType
    train = args.train
    data_type = args.dataType
    new_line = args.newLine
    train_data = args.trainData
    log_folder_name = args.log_folder_name
    attack = args.attack
    title = args.title

    # Print the arguments
    print(f"Title: {title}")
    print(f"Model Type: {model_type}")
    print(f"Train: {train}")
    print(f"Attack: {attack}")
    print(f"Training Data: {train_data}")
    print(f"Data Type: {data_type}")
    print(f"New Line: {new_line}")
    print(f"Log folder name: {log_folder_name}")

    results_report['Model Type'] = model_type
    results_report['Train'] = train
    results_report['Training Data'] = train_data
    results_report['Data Type'] = data_type
    results_report['New Line'] = new_line
    results_report['Log Folder Name'] = log_folder_name
    results_report['Title'] = title
    results_report['Attack'] = attack

    Run.run(model_type, train, data_type, new_line, train_data, log_folder_name, attack, title)

if __name__ == "__main__":
    main()
