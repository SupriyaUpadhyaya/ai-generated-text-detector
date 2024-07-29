import yaml
from datasets import load_dataset, DatasetDict, concatenate_datasets
from textattack.datasets import Dataset as TextAttackDataset

class AttackDataset:
    def __init__(self, yaml_path='config/dataset_paths.yaml'):
        self.yaml_path = yaml_path

    def getDataset(self):
        jsonPaths, col_names = self.getJsonPath()
        print(jsonPaths)

        dataset = self.load_and_merge_datasets(jsonPaths, col_names)
        return dataset

    def getJsonPath(self):
        with open(self.yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        test_paths = {}
        col_names = {}
        for key in config:
            paths = config[key]
            test_paths[key] = paths.get("test")
            col_names[key] = {"human_text_column": paths.get("human_text_column"), "machine_text_column": paths.get("machine_text_column")}
        return test_paths, col_names
        
    def load_and_process_jsonl_dataset(self, file_path, human_text_column, machine_text_column, human_label, machine_label):
        # Load the dataset
        dataset = load_dataset('json', data_files={'data': file_path})['data']
        dataset = dataset.select(range(2))
        
        # Functions to process human text and machine text
        def process_human_example(example):
            return {'text': example[human_text_column], 'label': human_label}

        def process_machine_example(example):
            return {'text': example[machine_text_column], 'label': machine_label}

        # Process datasets
        human_dataset = dataset.map(process_human_example, remove_columns=dataset.column_names)
        machine_dataset = dataset.map(process_machine_example, remove_columns=dataset.column_names)

        return human_dataset, machine_dataset

    def load_and_merge_datasets(self, jsonPaths, col_names):
        # Load and process each dataset split
        dataset = {}
        for path in jsonPaths:
            test_human_dataset, test_machine_dataset = self.load_and_process_jsonl_dataset(jsonPaths[path], col_names[path]["human_text_column"], col_names[path]["machine_text_column"], 0, 1)
            test_dataset = concatenate_datasets([test_human_dataset, test_machine_dataset])
            X_test = list(test_dataset["text"])
            y_test = list(test_dataset["label"])
            textattack_dataset = TextAttackDataset(list(zip(X_test, y_test)))
            #textattack_dataset = TextAttackDataset(list(test_dataset))
            #print("Here is the dataset!")
            #print(list(test_dataset))
            dataset[path]=textattack_dataset

        # Create a DatasetDict with all splits
        dataset_dict = DatasetDict(dataset)

        return dataset_dict
