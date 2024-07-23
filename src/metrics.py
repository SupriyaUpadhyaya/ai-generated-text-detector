import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets

log_dir='./logs'
writer = SummaryWriter(log_dir)

class Metrics:
    @staticmethod
    def compute_metrics(eval_pred):
        """
        Computes accuracy, precision, recall, and F1 score for the evaluation.
        """
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Log metrics to TensorBoard
        writer.add_scalar('eval/accuracy', accuracy)
        writer.add_scalar('eval/precision', precision)
        writer.add_scalar('eval/recall', recall)
        writer.add_scalar('eval/f1', f1)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def plot_metrics(trainer, path):
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
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('eval/loss', eval_loss, epoch)

        # Plot training and evaluation loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss', marker='o')
        if eval_losses:
            plt.plot(epochs[:len(eval_losses)], eval_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plot_path = f'./Training_and_Validation_Loss_{epoch}.png'
        plt.savefig(plot_path)
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
        plt.close()

    @staticmethod
    def plot_confusion_matrix(predictions, labels, name, path):
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
        plot_path = f'{path}/confusion_matrix_{name[0]}.png'
        plt.savefig(plot_path)
        plt.close()
        writer.add_figure(f'Confusion Matrix', plt.gcf())
