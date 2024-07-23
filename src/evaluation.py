from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from metrics import Metrics

class Evaluation:
    def __init__(self, model):
        self.tokenizer = RobertaTokenizer.from_pretrained('andreas122001/roberta-academic-detector')
        self.model = model
        self.metrics = Metrics(tokenizer=self.tokenizer)
    
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
    
    def evaluation(self, dataset_name='imdb', learning_rate=2e-5, train_batch_size=16, eval_batch_size=16, num_train_epochs=1, weight_decay=0.01):
        """
        Trains the model using the provided dataset.
        """
        # Load dataset
        dataset = load_dataset(dataset_name)
        tokenized_datasets = self.preprocess(dataset)

        # Define training arguments
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
            train_dataset=tokenized_datasets['train'],   # training dataset
            eval_dataset=tokenized_datasets['validation'], # evaluation dataset
            compute_metrics=Metrics.compute_metrics   # function to compute metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)

        # Test the model
        test_results = trainer.predict(tokenized_datasets['test'])
        print("Test results:", test_results.metrics)

        # Plot confusion matrix for test set
        test_predictions = test_results.predictions.argmax(axis=-1)
        test_labels = tokenized_datasets['test']['label']
        Metrics.plot_confusion_matrix(test_predictions, test_labels)
        Metrics.plot_metrics(training_args, trainer)
        Metrics.plot_confusion_matrix(test_predictions, test_labels)

# Usage Example
if __name__ == "__main__":
    trainer = Train()
    trainer.train()
