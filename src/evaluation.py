from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.metrics import Metrics
import yaml
import os
import torch
from safetensors.torch import save_file, load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation:
    def __init__(self, model_type, num_labels=2):
        with open('config/model.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model_name = config[model_type].get('pretrained')
        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            weights_path = f'./saved_weights/{model_type}/model.safetensors'
        
            # Load the model weights from the local directory
            if os.path.exists(weights_path):
                state_dict = load_file(weights_path)
                self.model.load_state_dict(state_dict)
                print(f"Model weights loaded from {weights_path}")
            else:
                print(f"No weights found at {weights_path}. Using the pre-trained model without additional weights.")
        
        self.metrics = Metrics()
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
        for type in datasets:
            print(f'************* Evaluation for {datasets[type]}')
            # Load dataset
            dataset = datasets[type]
            tokenized_datasets = self.preprocess(dataset)

            training_args = TrainingArguments(
            output_dir='./results',          # output directory
            evaluation_strategy="epoch",     # evaluate after each epoch
            learning_rate=learning_rate,     # learning rate
            per_device_train_batch_size=train_batch_size,   # batch size for training
            per_device_eval_batch_size=eval_batch_size,    # batch size for evaluation
            num_train_epochs=num_train_epochs,              # number of training epochs
            weight_decay=weight_decay,       # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=1,
            )

            # Initialize Trainer
            trainer = Trainer(
                model=self.model,                         # the instantiated model to train
                args=training_args,                       # training arguments
                compute_metrics=Metrics.compute_metrics   # function to compute metrics
            )


            test_results = trainer.predict(tokenized_datasets)
            print("Test results:", test_results.metrics)

            # Plot confusion matrix for test set
            test_predictions = test_results.predictions.argmax(axis=-1)
            test_labels = tokenized_datasets['label']
            Metrics.plot_confusion_matrix(test_predictions, test_labels, {datasets[type]})
