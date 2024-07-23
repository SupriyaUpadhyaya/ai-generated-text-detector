import argparse
from src.TrainingDataset import TrainingDataset

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

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print(f"Model Type: {args.modelType}")
    print(f"Train: {args.train}")
    print(f"Training Data: {args.trainData}")
    print(f"Data Type: {args.dataType}")
    print(f"New Line: {args.newLine}")

    # If train is true, create a TrainingDataset object and call getDataset
    if args.train:
        training_dataset = TrainingDataset()
        dataset = training_dataset.getDataset(trainData=args.trainData, dataType=args.dataType, newLine=args.newLine)
        print(f"Dataset obtained: {dataset}")

if __name__ == "__main__":
    main()
