import argparse
from src.utils.trainingDataset import TrainingDataset
from deep_learning_detector.training import Train
from deep_learning_detector.evaluation import Evaluation
from xgboost_detector.xgboost import TrainXGBoost
from xgboost_detector.xgboostEvaluation import EvaluationXGBoost
from src.utils.evaluationDataset import EvaluationDataset
from attack.attackDataset import AttackDataset
from attack.attack import Attack
import torch
from src.utils.results import Report

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

    if not attack:
        # If train is true, create a TrainingDataset object and call getDataset
        if train:
            training_dataset = TrainingDataset()
            dataset = training_dataset.getDataset(trainData=train_data, dataType=data_type, newLine=new_line)
            print(f"dataset : {dataset}")
            results_report['Training dataset obtained'] = dataset
            if model_type is not 'xgboost':
                training = Train(model_type, log_folder_name)
            else:
                training = TrainXGBoost(model_type, log_folder_name)
            training.train()
        
        evaluation_dataset = EvaluationDataset()
        dataset = evaluation_dataset.getDataset()
        print(f"dataset : {dataset}")
        results_report['Evaluation datasets obtained'] = dataset
        if model_type is not 'xgboost':
            evaluation = Evaluation(model_type, log_folder_name)
        else:
            evaluation = EvaluationXGBoost(model_type, log_folder_name)
        evaluation.evaluate(dataset)
    
    if attack:
        attack_dataset = AttackDataset()
        dataset = attack_dataset.getDataset()
        print(f"Dataset obtained: {dataset}")
        RobertaAttacker = Attack(model_type, log_folder_name)
        RobertaAttacker.attack(dataset['chatgpt_abstract_without'])

    Report.generateReport()

if __name__ == "__main__":
    main()
