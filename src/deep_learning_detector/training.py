from tkinter.filedialog import SaveFileDialog
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from src.utils.metrics import Metrics
import yaml
import torch
import os
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
from src.shared import results_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
    def __init__(self, model_type, log_folder_name, num_labels=2):
        print("model_type : ", model_type)
        self.model_type = model_type
        self.log_path = f'results/report/{self.model_type}/{log_folder_name}/'
        results_report['log_path']=self.log_path
        with open('config/model.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        print("self.config : ", self.config)
        model_name = self.config[model_type].get('pretrained')
        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        elif model_type == 'bloomz':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.metrics = Metrics(self.log_path)
        print(f'************************** Results PATH - {self.log_path} ***********************')

    def preprocess_function(self, examples):
        """
        Tokenizes the input examples.
        """
        if self.model_type == 'roberta':
            tokenized_input = self.tokenizer(examples['text'], padding='max_length', truncation=True)
        elif self.model_type == 'bloomz':
            tokenized_input = self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length = 512)
        return tokenized_input
    
    def preprocess(self, dataset):
        """
        Tokenizes the dataset using the provided tokenizer.
        """
        return dataset.map(lambda x: self.preprocess_function(x), batched=True)
    
    def train(self, dataset, learning_rate=2e-5, train_batch_size=16, eval_batch_size=16, num_train_epochs=3, weight_decay=0.01):
        """
        Trains the model using the provided dataset.
        """
        tokenized_datasets = self.preprocess(dataset)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f'{self.log_path}/results',          # output directory
            eval_strategy="epoch",     # evaluate after each epoch
            learning_rate=learning_rate,     # learning rate
            per_device_train_batch_size=train_batch_size,   # batch size for training
            per_device_eval_batch_size=eval_batch_size,    # batch size for evaluation
            num_train_epochs=num_train_epochs,              # number of training epochs
            weight_decay=weight_decay,       # strength of weight decay
            logging_dir=f'{self.log_path}/logs',            # directory for storing logs
            logging_steps=1,
            report_to="tensorboard"
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,                         # the instantiated model to train
            args=training_args,                       # training arguments
            train_dataset=tokenized_datasets['train'],   # training dataset
            eval_dataset=tokenized_datasets['validation'], # evaluation dataset
            compute_metrics=self.metrics.compute_metrics   # function to compute metrics
        )

        # Train the model
        trainer.train()

        # Save only the model weights in safetensors format
        weights_path = f'{self.log_path}/saved_weights'
        os.makedirs(weights_path, exist_ok=True)
        save_file(self.model.state_dict(), os.path.join(weights_path, 'model.safetensors'))

        self.config[self.model_type]['finetuned']= f'{weights_path}/model.safetensors'
        with open('config/model.yaml', 'w') as file:
            yaml.safe_dump(self.config, file)


        # Evaluate the model
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)
        results_report['Validation results'] = eval_results

        self.metrics.plot_metrics(trainer, f'{self.log_path}')

        # Test the model
        test_results = trainer.predict(tokenized_datasets['test'])
        print("Test results:", test_results.metrics)
        results_report['Test results'] = test_results.metrics

        # Plot confusion matrix for test set
        test_predictions = test_results.predictions.argmax(axis=-1)
        test_labels = tokenized_datasets['test']['label']
        self.metrics.compute_metrics((torch.tensor(test_results.predictions), torch.tensor(test_labels)))
        self.metrics.plot_confusion_matrix(test_predictions, test_labels, 'Test', self.log_path)

