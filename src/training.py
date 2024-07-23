from tkinter.filedialog import SaveFileDialog
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.metrics import Metrics
import yaml
import torch
import os
from safetensors.torch import save_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
    def __init__(self, model_type,log_path, num_labels=2):
        print("model_type : ", model_type)
        with open('config/model.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        print("self.config : ", self.config)
        model_name = self.config[model_type].get('pretrained')
        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.metrics = Metrics()
        self.model_type = model_type
        self.log_path = log_path

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
    
    def train(self, dataset, learning_rate=2e-5, train_batch_size=16, eval_batch_size=16, num_train_epochs=1, weight_decay=0.01):
        """
        Trains the model using the provided dataset.
        """
        tokenized_datasets = self.preprocess(dataset)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            eval_strategy="epoch",     # evaluate after each epoch
            learning_rate=learning_rate,     # learning rate
            per_device_train_batch_size=train_batch_size,   # batch size for training
            per_device_eval_batch_size=eval_batch_size,    # batch size for evaluation
            num_train_epochs=num_train_epochs,              # number of training epochs
            weight_decay=weight_decay,       # strength of weight decay
            logging_dir=f'{self.log_path}/{self.model_type}/logs',            # directory for storing logs
            logging_steps=1,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,                         # the instantiated model to train
            args=training_args,                       # training arguments
            train_dataset=tokenized_datasets['train'],   # training dataset
            eval_dataset=tokenized_datasets['validation'], # evaluation dataset
            compute_metrics=Metrics.compute_metrics   # function to compute metrics
        )

        # Train the model
        trainer.train()

        # Save only the model weights in safetensors format
        weights_path = f'{self.log_path}/{self.model_type}/saved_weights'
        self.config[self.model_type]['finetuned']=weights_path
        with open('config/model.yaml', 'w') as file:
            yaml.safe_dump(self.config, file)
        os.makedirs(weights_path, exist_ok=True)
        save_file(self.model.state_dict(), os.path.join(weights_path, 'model.safetensors'))


        # Evaluate the model
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)

        # Test the model
        test_results = trainer.predict(tokenized_datasets['test'])
        print("Test results:", test_results.metrics)

        # Plot confusion matrix for test set
        test_predictions = test_results.predictions.argmax(axis=-1)
        test_labels = tokenized_datasets['test']['label']
        Metrics.plot_confusion_matrix(test_predictions, test_labels, 'Test', f'{self.log_path}/{self.model_type}/logs')

