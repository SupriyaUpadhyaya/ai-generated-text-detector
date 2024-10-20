import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import shap
from src.utils.metrics import Metrics
import torch
from src.xgboost_detector.featureExtractor import FeatureExtractor
from src.shared import results_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainXGBoost:
    def __init__(self, model_type,log_folder_name):
        print("model_type : ", model_type)
        self.model_type = model_type
        self.log_path = f'results/report/{self.model_type}/{log_folder_name}/'
        results_report['log_path']=self.log_path
        self.metrics = Metrics(f'{self.log_path}/{self.model_type}/logs')
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
        print(f'************************** LOG PATH - {self.log_path} ***********************')

    def train(self, dataset):
        X_train = FeatureExtractor.getFeatures(dataset['train']['text'])
        X_val = FeatureExtractor.getFeatures(dataset['validation']['text'])
        X_test = FeatureExtractor.getFeatures(dataset['test']['text'])
        y_train = dataset['train']['label']
        y_val = dataset['validation']['label']
        y_test = dataset['test']['label']
        
        plt.figure(figsize=(10, 6))
        
        evalset = [(X_train, y_train), (X_val,y_val)]

        importances_gain = pd.DataFrame()
        importances_weight = pd.DataFrame()
        importances_cover = pd.DataFrame()
        importances_total_gain = pd.DataFrame()

        self.xgb_classifier.fit(X_train, y_train,
                        eval_set=evalset,
                        verbose=False)
        results = self.xgb_classifier.evals_result()
        yhat = self.xgb_classifier.predict(X_test)
        score = accuracy_score(y_test, yhat)
        print('Accuracy: %.3f' % score)
        results_report['Training accuracy'] = score

        plt.plot(results['validation_0']['logloss'], label='train')
        plt.plot(results['validation_1']['logloss'], label='validation')
        values = list(X_train.columns.values)
        title = ' '.join(values)
        plt.title('Loss vs. Epoch for ')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

        booster = self.xgb_classifier.get_booster()
        importance_score = booster.get_score(importance_type='gain')
        importance_frame = pd.DataFrame({'Importance': list(importance_score.values()), 
                                        'Feature': list(importance_score.keys())})
        importances_gain = pd.concat([importances_gain, 
                                        importance_frame],
                                        axis=0,
                                        sort=False)

        importance_score = booster.get_score(importance_type='weight')
        importance_frame = pd.DataFrame({'Importance': list(importance_score.values()), 'Feature': list(importance_score.keys())})
        importances_weight = pd.concat([importances_weight, importance_frame], axis=0, sort=False)

        importance_score = booster.get_score(importance_type='cover')
        importance_frame = pd.DataFrame({'Importance': list(importance_score.values()), 'Feature': list(importance_score.keys())})
        importances_cover = pd.concat([importances_cover, importance_frame], axis=0, sort=False)

        importance_score = booster.get_score(importance_type='total_gain')
        importance_frame = pd.DataFrame({'Importance': list(importance_score.values()), 'Feature': list(importance_score.keys())})
        importances_total_gain = pd.concat([importances_total_gain, importance_frame], axis=0, sort=False)

        mean_total_gain = importances_total_gain[['Importance', 'Feature']].groupby('Feature').mean()
        mean_total_gain = mean_total_gain.reset_index()
        plt.figure(figsize=(17, 8))
        sns.barplot(x='Importance', y='Feature', 
                                    data=mean_total_gain.sort_values('Importance',
                                    ascending=False), 
                                    palette='gray')
        plt.savefig(f'{self.log_path}_importance.png')
        self.xgb_classifier.save_model('save_models/xgboost_model.json')

        self.performance_test(X_test, y_test)

        return self.xgb_classifier, score
    
    def performance_test(self, X_test_list, y_test_list):
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
        plt.savefig(f'{self.log_path}/confusion_matrix_test.png')
        f1score = f1_score(y_test_list, y_pred, zero_division=1.0)
        precision_recall_fscore = precision_recall_fscore_support(y_test_list, y_pred, zero_division=1.0)
        print("F1 score : ", f1score)
        print("precision_recall_fscore : ", precision_recall_fscore)
        results_report["precision_recall_fscore"]= precision_recall_fscore
        results_report['F1 score']=f1score

