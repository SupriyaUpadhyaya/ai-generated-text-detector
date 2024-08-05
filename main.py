import argparse
from src.utils.trainingDataset import TrainingDataset
from src.deep_learning_detector.training import Train
from src.deep_learning_detector.evaluation import Evaluation
from src.xgboost_detector.xgboost import TrainXGBoost
from src.xgboost_detector.xgboostEvaluation import EvaluationXGBoost
from src.utils.evaluationDataset import EvaluationDataset
from src.attack.attackDataset import AttackDataset
from src.attack.attack import Attack
from src.attack.attack_xgboost import AttackXGBoost
import torch
from src.shared import results_report
from src.utils.results import Report
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some input arguments.")

    # Add arguments
    parser.add_argument("--modelType", type=str, choices=["roberta", "bloomz", "xgboost"], default="roberta", 
                        help="Type of the model: roberta, bloomz, or xgboost")
    # parser.add_argument("--train", type=lambda x: (str(x).lower() == 'true'), default=True,
    #                     help="Train the model: true or false")
    parser.add_argument("--trainData", type=str, choices=["chatgpt", "bloomz", "cohere", "flat5", "davinci", "llama3"], default="chatgpt",
                        help="Training data source: chatgpt, bloomz, cohere, flat5, davinci or llama3")
    parser.add_argument("--dataType", type=str, choices=["abstract", "wiki", "ml"], default="abstract",
                        help="Type of data: abstract, wiki, or ml")
    parser.add_argument("--task", type=str, choices=["evalPretrained", "finetuneAndEval", "attack", "trainOnSubset"], default="None",
                        help="Type of data: abstract, wiki, or ml")
    parser.add_argument("--newLine", type=str, choices=["with", "without"], default="without",
                        help="Include new lines in the output: with or without")
    parser.add_argument('--log_folder_name', type=str, required=True, help='Log folder name')
    # parser.add_argument("--attack", type=lambda x: (str(x).lower() == 'true'), default=False,
    #                     help="Run attack on the model: true or false")
    parser.add_argument('--title', type=str, required=True, help='Title')
    # parser.add_argument("--TrainOnSubset", type=lambda x: (str(x).lower() == 'true'), default=False,
    #                     help="Train the model with subset of training data: true or false")
    parser.add_argument('--percentage', type=float, help='percentage')

    # Parse the arguments
    args = parser.parse_args()
    model_type = args.modelType
    task = args.task
    data_type = args.dataType
    new_line = args.newLine
    train_data = args.trainData
    log_folder_name = args.log_folder_name
    title = args.title
    percentage = args.percentage

    # Print the arguments
    print(f"Title: {title}")
    print(f"Model Type: {model_type}")
    print(f"Task: {task}")
    print(f"Training Data: {train_data}")
    print(f"Data Type: {data_type}")
    print(f"New Line: {new_line}")
    print(f"Log folder name: {log_folder_name}")
    print(f"percentage: {percentage}")

    results_report['Model Type'] = model_type
    results_report['Task'] = task
    results_report['Training Data'] = train_data
    results_report['Data Type'] = data_type
    results_report['New Line'] = new_line
    results_report['Log Folder Name'] = log_folder_name
    results_report['Title'] = title
    results_report['percentage'] = percentage

    # If task is finetuneAndEval, create a TrainingDataset object and call getDataset
    if task == "finetuneAndEval":
        training_dataset = TrainingDataset()
        dataset = training_dataset.getDataset(trainData=train_data, dataType=data_type, newLine=new_line)
        print(f"dataset : {dataset}")
        if model_type != 'xgboost':
            training = Train(model_type, log_folder_name)
        else:
            training = TrainXGBoost(model_type, log_folder_name)
        training.train(dataset)
    
        evaluation_dataset = EvaluationDataset()
        dataset = evaluation_dataset.getDataset()
        print(f"dataset : {dataset}")

        if model_type != 'xgboost':
            evaluation = Evaluation(model_type, log_folder_name)
        else:
            evaluation = EvaluationXGBoost(model_type, log_folder_name)
        evaluation.evaluate(dataset)

        Report.generateReport()
        with open(f'{results_report["log_path"]}/results_report.json', 'w') as json_file:
            json.dump(results_report, json_file, indent=4)
    
    elif task == "evalPretrained":
        evaluation_dataset = EvaluationDataset()
        dataset = evaluation_dataset.getDataset()
        print(f"dataset : {dataset}")

        if model_type != 'xgboost':
            evaluation = Evaluation(model_type, log_folder_name)
        else:
            evaluation = EvaluationXGBoost(model_type, log_folder_name)
        evaluation.evaluate(dataset)

        Report.generateReport()
        with open(f'{results_report["log_path"]}/results_report.json', 'w') as json_file:
            json.dump(results_report, json_file, indent=4)

    elif task == "attack":
        attack_dataset = AttackDataset()
        dataset = attack_dataset.getDataset()
        print(f"Dataset obtained: {dataset}")
        print("dataset : ", f'{train_data}_{data_type}_{new_line}')
        if model_type != 'xgboost':
            DLAttacker = Attack(model_type, log_folder_name)
            DLAttacker.attack(dataset[f'{train_data}_{data_type}_{new_line}'])
        else:
            XGBoostAttacker = AttackXGBoost(model_type, log_folder_name)
            XGBoostAttacker.attack(dataset[f'{train_data}_{data_type}_{new_line}'])
        
        Report.generateReport()
        with open(f'{results_report["log_path"]}/results_report.json', 'w') as json_file:
            json.dump(results_report, json_file, indent=4)

    elif task == "trainOnSubset":
        training_dataset = TrainingDataset()
        name_suffix = str(int(percentage * 100))
        if model_type != 'xgboost':
            training = Train(model_type, log_folder_name)
        else:
            training = TrainXGBoost(model_type, log_folder_name)
        
        print(f'Percentage of training data used :', {(percentage)})
        dataset = training_dataset.getDataset(trainData=train_data, dataType=data_type, newLine=new_line, subset=(percentage))
        print(f"dataset : {dataset}")
        training.train(dataset)

        evaluation_dataset = EvaluationDataset()
        dataset = evaluation_dataset.getDataset()
        print(f"dataset : {dataset}")

        if model_type != 'xgboost':
            evaluation = Evaluation(model_type, log_folder_name)
        else:
            evaluation = EvaluationXGBoost(model_type, log_folder_name)
        evaluation.evaluate(dataset)

        Report.generateReport()
        with open(f'{results_report["log_path"]}/results_report.json', 'w') as json_file:
            json.dump(results_report, json_file, indent=4)


        Report.generateReport()
        with open(f'{results_report["log_path"]}/results_report.json', 'w') as json_file:
            json.dump(results_report, json_file, indent=4)
        
    else:
        print('Missing inline input argument: specify input --task argument value from ["evalPretrained", "finetuneAndEval", "attack", "trainOnSubset"]')

if __name__ == "__main__":
    main()
