from src.utils.trainingDataset import TrainingDataset
from src.deep_learning_detector.training import Train
from src.deep_learning_detector.evaluation import Evaluation
from src.xgboost_detector.xgboost import TrainXGBoost
from src.xgboost_detector.xgboostEvaluation import EvaluationXGBoost
from src.utils.evaluationDataset import EvaluationDataset
from src.attack.attackDataset import AttackDataset
from src.attack.attack import Attack
from src.utils.results import Report
from main import results_report

class Run:
     @staticmethod
     def execution(model_type, train, data_type, new_line, train_data, log_folder_name, attack, title):
        results_report['Title'] = title
        if not attack:
            # If train is true, create a TrainingDataset object and call getDataset
            if train:
                training_dataset = TrainingDataset()
                dataset = training_dataset.getDataset(trainData=train_data, dataType=data_type, newLine=new_line)
                print(f"dataset : {dataset}")
                results_report['Training dataset obtained'] = dataset
                if model_type != 'xgboost':
                    training = Train(model_type, log_folder_name)
                else:
                    training = TrainXGBoost(model_type, log_folder_name)
                training.train()
            
            evaluation_dataset = EvaluationDataset()
            dataset = evaluation_dataset.getDataset()
            print(f"dataset : {dataset}")
            results_report['Evaluation datasets obtained'] = dataset
            if model_type != 'xgboost':
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