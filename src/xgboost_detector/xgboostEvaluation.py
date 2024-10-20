from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.utils.metrics import Metrics
import yaml
import os
import torch
from safetensors.torch import load_file
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import seaborn as sns
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import shap
from src.xgboost_detector.featureExtractor import FeatureExtractor
from src.shared import results_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluationXGBoost:
    def __init__(self, model_type, log_folder_name, num_labels=2):
        self.model_type = model_type
        self.log_path = self.log_path = f'results/report/{self.model_type}/{log_folder_name}/'
        results_report['log_path']=self.log_path
        with open('config/model.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        lr = 0.01
        weight = 4.5
        self.xgb_classifier = XGBClassifier(n_estimators=500,
                                    use_label_encoder=False,
                                    eval_metric="logloss",
                                    early_stopping_rounds=5,
                                    n_jobs=-1,
                                    eta=lr,
                                    reg_lambda=1,
                                    min_child_weight=weight)
        weights_path = self.config[model_type].get('finetuned')
        print('weights_path :', weights_path)
        # Load the model weights from the local directory
        if os.path.exists(weights_path):
            self.xgb_classifier.load_model(weights_path)
            print(f"Model weights loaded from {weights_path}")
        else:
            print(f"No weights found at {weights_path}. Using the pre-trained model without additional weights.")
    
        self.metrics = Metrics(self.log_path)
    
    def evaluate(self, datasets):
        for dstype in datasets:
            print(f'************* Evaluation for {dstype} *************')
            # Load dataset
            dataset = datasets[dstype]
            X_test = FeatureExtractor.getFeatures(dataset['test']['text'])
            y_test = dataset['test']['label']
            self.performance_test(X_test, y_test, dstype)
            
    
    def performance_test(self, X_test_list, y_test_list, dstype):
        y_pred = self.xgb_classifier.predict(X_test_list)
        # Compute confusion matrix
        cm = confusion_matrix(y_test_list, y_pred)

        # Plot confusion matrix using seaborn heatmap
        plt.figure(figsize=(8, 6))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Human', 'Machine'], yticklabels=['Human', 'Machine'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show(block=False)
        explainer = shap.Explainer(self.xgb_classifier)
        shap_values = explainer(X_test_list)
        shap.summary_plot(shap_values, X_test_list)
        shap.plots.heatmap(shap_values)
        plt.savefig(f'{self.log_path}/{dstype}_confusion_matrix.png')
        f1score = f1_score(y_test_list, y_pred, zero_division=1.0)
        precision_recall_fscore = precision_recall_fscore_support(y_test_list, y_pred, zero_division=1.0)
        print(f"F1 score {dstype}: ", f1score)
        print(f"precision_recall_fscore {dstype}: ", precision_recall_fscore)
        results_report[f"precision_recall_fscore {dstype}"]= precision_recall_fscore
        results_report[f'F1 score {dstype}']=f1score
