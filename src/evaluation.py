from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.metrics import Metrics
import yaml
import os
import torch
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation:
    def __init__(self, model_type, log_path, num_labels=2):
        self.model_type = model_type
        self.log_path = log_path
        with open('config/model.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        model_name = self.config[model_type].get('pretrained')
        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            weights_path = self.config[model_type].get('finetuned')
            print('weights_path :', weights_path)
            # Load the model weights from the local directory
            if os.path.exists(weights_path):
                state_dict = load_file(weights_path)
                self.model.load_state_dict(state_dict)
                print(f"Model weights loaded from {weights_path}")
            else:
                print(f"No weights found at {weights_path}. Using the pre-trained model without additional weights.")
        
        self.metrics = Metrics(f'{self.log_path}/{self.model_type}/logs')
    
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
        for type in datasets:
            print(f'************* Evaluation for {type} *************')
            # Load dataset
            dataset = datasets[type]
            tokenized_datasets = self.preprocess(dataset)

            training_args = TrainingArguments(
            output_dir=f'{self.log_path}/{self.model_type}/results',          # output directory
            per_device_train_batch_size=train_batch_size,   # batch size for training
            per_device_eval_batch_size=eval_batch_size,    # batch size for evaluation
            logging_dir=f'{self.log_path}/{self.model_type}/logs',            # directory for storing logs
            logging_steps=1,
            )

            # Initialize Trainer
            trainer = Trainer(
                model=self.model,                         # the instantiated model to train
                args=training_args,                       # training arguments
                compute_metrics=self.metrics.compute_metrics   # function to compute metrics
            )


            test_results = trainer.predict(tokenized_datasets)
            print("Test results:", test_results.metrics)

            # Plot confusion matrix for test set
            test_predictions = test_results.predictions.argmax(axis=-1)
            test_labels = tokenized_datasets['label']
            self.metrics.plot_confusion_matrix(test_predictions, test_labels, f'{type}', f'{self.log_path}/{self.model_type}/logs')
