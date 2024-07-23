from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.metrics import Metrics
import yaml
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation:
    def __init__(self, model_type, num_labels=2):
        with open('config/model.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model_name = config[model_type].get('pretrained')
        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            weights_path = f'./saved_weights/{model_type}'        
            # Load the model weights from the local directory
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(os.path.join(weights_path, 'pytorch_model.bin')))
                print(f"Model weights loaded from {weights_path}")
            else:
                print(f"No weights found at {weights_path}. Using the pre-trained model without additional weights.")
        self.metrics = Metrics(tokenizer=self.tokenizer)
        self.model_type = model_type
    
    def preprocess_function(self, examples):
        """
        Tokenizes the input examples.
        """
        return self.tokenizer(examples['text'], padding='max_length', truncation=True)
    
    def preprocess(self, dataset):
        """
        Tokenizes the dataset using the provided tokenizer.
        """
        return dataset.map(lambda x: self.preprocess_function(x), batched=True)
    
    def evaluate(self, datasets, learning_rate=2e-5, train_batch_size=16, eval_batch_size=16, num_train_epochs=1, weight_decay=0.01):
        for dataset in datasets:

            # Load dataset
            tokenized_datasets = self.preprocess(dataset)
            test_results = trainer.predict(tokenized_datasets['test'])
            print("Test results:", test_results.metrics)

            # Plot confusion matrix for test set
            test_predictions = test_results.predictions.argmax(axis=-1)
            test_labels = tokenized_datasets['test']['label']
            Metrics.plot_confusion_matrix(test_predictions, test_labels)
            Metrics.plot_confusion_matrix(test_predictions, test_labels)
