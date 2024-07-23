import argparse
from src.trainingDataset import TrainingDataset
from src.training import Train
from src.evaluation import Evaluation
from src.evaluationDataset import EvaluationDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model_type = args.modelType
    train = args.train
    data_type = args.dataType
    new_line = args.newLine
    train_data = args.trainData
    # Print the arguments
    print(f"Model Type: {args.modelType}")
    print(f"Train: {args.train}")
    print(f"Training Data: {args.trainData}")
    print(f"Data Type: {args.dataType}")
    print(f"New Line: {args.newLine}")

    # If train is true, create a TrainingDataset object and call getDataset
    if train:
        training_dataset = TrainingDataset()
        dataset = training_dataset.getDataset(trainData=train_data, dataType=data_type, newLine=new_line)
        print(f"Dataset obtained: {dataset}")
        training = Train(model_type)
        model_path = training.train(dataset)
    
    evaluation_dataset = EvaluationDataset()
    dataset = evaluation_dataset.getDataset()
    print(f"Dataset obtained: {dataset}")
    evaluation = Evaluation(model_type)
    evaluation.evaluate(dataset)

if __name__ == "__main__":
    main()
