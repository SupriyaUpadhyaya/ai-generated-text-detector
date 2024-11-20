import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets
import pandas as pd

class Metrics:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def compute_metrics(self, eval_pred):
        """
        Computes accuracy, precision, recall, and F1 score for the evaluation.
        """
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Log metrics to TensorBoard
        self.writer.add_scalar('eval/accuracy', accuracy)
        self.writer.add_scalar('eval/precision', precision)
        self.writer.add_scalar('eval/recall', recall)
        self.writer.add_scalar('eval/f1', f1)
        self.writer.add_scalar('eval/sensitivity', sensitivity)
        self.writer.add_scalar('eval/specificity', specificity)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    
    def plot_metrics(self, trainer, path):
        """
        Plots training loss and accuracy curves.
        """
        # Extract metrics from trainer
        logs = trainer.state.log_history
        epochs = [log['epoch'] for log in logs if 'epoch' in log]
        train_losses = [log['loss'] for log in logs if 'loss' in log]
        eval_losses = [log['eval_loss'] for log in logs if 'eval_loss' in log]
        train_accuracies = [log['accuracy'] for log in logs if 'accuracy' in log]
        eval_accuracies = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]

        # Ensure that the lengths match
        if len(epochs) != len(train_losses):
            print(f"Mismatch in lengths: epochs ({len(epochs)}) vs train_losses ({len(train_losses)})")

        # Log training and evaluation loss to TensorBoard
        for epoch, train_loss, eval_loss in zip(epochs, train_losses, eval_losses):
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('eval/loss', eval_loss, epoch)

        # Plot training and evaluation loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss', marker='o')
        if eval_losses:
            plt.plot(epochs[:len(eval_losses)], eval_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plot_path = f'{path}//Training_and_Validation_Loss.png'
        plt.savefig(plot_path)
        self.writer.add_figure(plot_path, plt.gcf())
        plt.close()

        # Plot evaluation accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(epochs[:len(train_accuracies)], train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(epochs[:len(eval_accuracies)], eval_accuracies, label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Vs Validation Accuracy')
        plt.legend()
        plt.savefig(f'{path}/Training_Validationn_Accuracy.png')
        self.writer.add_figure(f'{path}/Training_Validationn_Accuracy.png', plt.gcf())
        plt.close()

    def plot_confusion_matrix(self, predictions, labels, name, path):
        """
        Plots the confusion matrix for the test set.
        """
        cm = confusion_matrix(labels, predictions)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cmn, annot=True, fmt='.2f',
                    xticklabels=['Class 0 - Human', 'Class 1 - Machine'], yticklabels=['Class 0 - Human', 'Class 1 - Machine'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plot_path = f'{path}/confusion_matrix_{name}.png'
        plt.savefig(plot_path)
        self.writer.add_figure(f'{path}/confusion_matrix_{name}.png', plt.gcf())
        plt.close()

    def qualitative_analysis(self, text, X_test, y_test, y_pred, name, path):
        print("qualitative_analysis")
        print("y_test : ", len(y_test))
        print("y_pred : ", len(y_pred))
        print("text : ", len(text))
        print("X_test : ", len(X_test))
        print("X_test : ", X_test)

        X_test_df = pd.DataFrame(X_test)

        mis_cls = []
        mis_cls_feat = []
        count = 0
        for text_sample, truth, prediction in zip(text, y_test, y_pred):
            print("prediction :", prediction)
            print("truth :", truth)
            if prediction != truth:
                mis_cls.append({
                    "text": text_sample,
                    "truth": truth,
                    "prediction": prediction
                })
                mis_cls_feat.append[X_test_df.iloc[count, :]]
                count += 1
        print("count : ", count)

        correct_cls = []
        correct_cls_feat = []
        count = 0
        for text_sample, truth, prediction in zip(text, y_test, y_pred):
            print("prediction :", prediction)
            print("truth :", truth)
            if prediction == truth:
                correct_cls.append({
                    "text": text_sample,
                    "truth": truth,
                    "prediction": prediction
                })
                correct_cls_feat.append[X_test_df.iloc[count, :]]
                count += 1
        print("count : ", count)

        # Converting to DataFrame
        print("mis_cls : ", len(mis_cls))
        print("correct_cls : ", len(correct_cls))
        print("mis_cls_feat : ", len(mis_cls_feat))
        print("correct_cls_feat : ", len(correct_cls_feat))
        mis_cls_df = pd.DataFrame(mis_cls)
        correct_cls_df = pd.DataFrame(correct_cls)
        mis_cls_feat_df = pd.DataFrame(mis_cls_feat)
        correct_cls_feat_df = pd.DataFrame(correct_cls_feat)

        # Save to CSV
        mis_cls_df.to_csv(f'{path}/misclassified_samples_{name}.csv', index=False)
        correct_cls_df.to_csv(f'{path}/correct_classified_samples_{name}.csv', index=False)
        mis_cls_feat_df.to_csv(f'{path}/misclassified_samples_features_{name}.csv', index=False)
        correct_cls_feat_df.to_csv(f'{path}/correct_classified_samples_features_{name}.csv', index=False)
