import yaml
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np
from src.shared import results_report

class TrainingDataset:
    def __init__(self, yaml_path='config/dataset_paths_preprocessed.yaml'):
        self.yaml_path = yaml_path

    def getDataset(self, trainData, dataType, newLine, subset=None):
        jsonPaths = self.getJsonPath(trainData, dataType, newLine)
        print(jsonPaths)
        results_report['Train input path'] = jsonPaths['train']
        results_report['Test input path'] = jsonPaths['test']
        results_report['Validation input path'] = jsonPaths['validation']
        results_report['Human text column name'] = jsonPaths['human_text_column']
        results_report['Machine text column name'] = jsonPaths['machine_text_column']

        dataset = self.load_and_merge_datasets(jsonPaths['train'], jsonPaths['test'], jsonPaths['validation'], jsonPaths['human_text_column'], jsonPaths['machine_text_column'], subset)
        return dataset

    def getJsonPath(self, trainData, dataType, newLine):
        with open(self.yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        key = f"{trainData}_{dataType}_{newLine}"
        if key in config:
            paths = config[key]
            return {
                "train": paths.get("train"),
                "validation": paths.get("validation"),
                "test": paths.get("test"),
                "human_text_column": paths.get("human_text_column"),
                "machine_text_column": paths.get("machine_text_column")
            }
        else:
            raise ValueError(f"No data paths found for key: {key}")
        
    def load_and_process_jsonl_dataset(self, file_path, human_text_column, machine_text_column, human_label, machine_label, subset=None):
        # Load the dataset
        dataset = load_dataset('json', data_files={'data': file_path})['data']

        if subset != None:
            value = int(len(dataset) * subset)
            print("dataset len : ", len(dataset))
            dataset = dataset.select(range(value))
        # Functions to process human text and machine text
        def process_human_example(example):
            return {'text': example[human_text_column], 'label': human_label}

        def process_machine_example(example):
            return {'text': example[machine_text_column], 'label': machine_label}

        # Process datasets
        human_dataset = dataset.map(process_human_example, remove_columns=dataset.column_names)
        machine_dataset = dataset.map(process_machine_example, remove_columns=dataset.column_names)

        return human_dataset, machine_dataset

    def load_and_merge_datasets(self, train_file, test_file, validation_file, human_text_column, machine_text_column, subset):
        # Load and process each dataset split
        train_human_dataset, train_machine_dataset = self.load_and_process_jsonl_dataset(train_file, human_text_column, machine_text_column, 0, 1, subset)
        test_human_dataset, test_machine_dataset = self.load_and_process_jsonl_dataset(test_file, human_text_column, machine_text_column, 0, 1)
        validation_human_dataset, validation_machine_dataset = self.load_and_process_jsonl_dataset(validation_file, human_text_column, machine_text_column, 0, 1)

        # Concatenate the human and machine text datasets for each split
        train_dataset = concatenate_datasets([train_human_dataset, train_machine_dataset])
        test_dataset = concatenate_datasets([test_human_dataset, test_machine_dataset])
        validation_dataset = concatenate_datasets([validation_human_dataset, validation_machine_dataset])

        # Create a DatasetDict with all splits
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'validation': validation_dataset
        })

        return dataset_dict
