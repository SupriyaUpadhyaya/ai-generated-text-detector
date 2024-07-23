import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import RobertaTokenizer

class Metrics:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else RobertaTokenizer.from_pretrained('roberta-base')

    @staticmethod
    def compute_metrics(eval_pred):
        """
        Computes accuracy, precision, recall, and F1 score for the evaluation.
        """
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted')
        }

    @staticmethod
    def plot_metrics(trainer):
        """
        Plots training loss and accuracy curves.
        """
        # Extract metrics from trainer
        logs = trainer.state.log_history
        epochs = [log['epoch'] for log in logs if 'epoch' in log]
        train_losses = [log['loss'] for log in logs if 'loss' in log]
        eval_losses = [log['eval_loss'] for log in logs if 'eval_loss' in log]

        # Ensure that the lengths match
        if len(epochs) != len(train_losses):
            print(f"Mismatch in lengths: epochs ({len(epochs)}) vs train_losses ({len(train_losses)})")

        # Plot training and evaluation loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss', marker='o')
        if eval_losses:
            plt.plot(epochs[:len(eval_losses)], eval_losses, label='Eval Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        plt.show()

        # Plot evaluation accuracy
        eval_accuracies = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]
        plt.figure(figsize=(12, 6))
        plt.plot(epochs[:len(eval_accuracies)], eval_accuracies, label='Eval Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(predictions, labels):
        """
        Plots the confusion matrix for the test set.
        """
        cm = confusion_matrix(labels, predictions)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cmn, annot=True, fmt='.2f',
                    xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
