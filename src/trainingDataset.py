import yaml
from datasets import load_dataset, DatasetDict, concatenate_datasets

class TrainingDataset:
    def __init__(self, yaml_path='config/dataset_paths.yaml'):
        self.yaml_path = yaml_path

    def getDataset(self, trainData, dataType, newLine):
        jsonPaths = self.getJsonPath(trainData, dataType, newLine)
        print(jsonPaths)
        human_text_column = 'human_text'
        machine_text_column = 'machine_text'

        dataset = self.load_and_merge_datasets(jsonPaths['train'], jsonPaths['test'], jsonPaths['validation'], jsonPaths['human_text_column'], jsonPaths['machine_text_column'])
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
        
    def load_and_process_jsonl_dataset(self, file_path, human_text_column, machine_text_column, human_label, machine_label):
        # Load the dataset
        dataset = load_dataset('json', data_files={'data': file_path})['data']
        dataset = dataset.select(range(20))
        # Functions to process human text and machine text
        def process_human_example(example):
            return {'text': example[human_text_column], 'label': human_label}

        def process_machine_example(example):
            return {'text': example[machine_text_column], 'label': machine_label}

        # Process datasets
        human_dataset = dataset.map(process_human_example, remove_columns=dataset.column_names)
        machine_dataset = dataset.map(process_machine_example, remove_columns=dataset.column_names)

        return human_dataset, machine_dataset

    def load_and_merge_datasets(self, train_file, test_file, validation_file, human_text_column, machine_text_column):
        # Load and process each dataset split
        train_human_dataset, train_machine_dataset = self.load_and_process_jsonl_dataset(train_file, human_text_column, machine_text_column, 0, 1)
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
