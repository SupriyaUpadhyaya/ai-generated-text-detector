from src.utils.metrics import Metrics
import yaml
import torch
import os
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
from src.attack.xgboostWrapper import XGBoostWrapper
import numpy as np
from textattack.attack_recipes import PWWSRen2019, Pruthi2019, DeepWordBugGao2018
from src.attack.attack_recipes import PWWSRen2019_threshold
import textattack
from textattack import Attacker
from safetensors.torch import load_file
from src.shared import results_report
from xgboost import XGBClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttackXGBoost:
    def __init__(self, model_type,log_folder_name, num_labels=2):
        print("model_type : ", model_type)
        self.model_type = model_type
        self.log_path = f'results/report/{self.model_type}/{log_folder_name}/'
        results_report['log_path']=self.log_path
        with open('config/model.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        print("self.config : ", self.config)
        lr = 0.01
        weight = 4.5
        self.xgb_classifier = XGBClassifier(n_estimators=500,
                                    use_label_encoder=False,
                                    eval_metric="logloss",
                                    early_stopping_rounds=5,
                                    n_jobs=-1,
                                    eta=lr,
                                    reg_lambda=1,
                                    min_child_weight=weight)
        weights_path = self.config[model_type].get('finetuned')
        print('weights_path :', weights_path)
        # Load the model weights from the local directory
        if os.path.exists(weights_path):
            self.xgb_classifier.load_model(weights_path)
            print(f"Model weights loaded from {weights_path}")
        else:
            print(f"No weights found at {weights_path}. Using the pre-trained model without additional weights.")
        self.model_wrapper = XGBoostWrapper(self.xgb_classifier)
        self.metrics = Metrics(self.log_path)
        self.attack_recipe = ['pwws']
        print(f'************************** LOG PATH - {self.log_path} ***********************')
    
    def attack(self, dataset):
        """
        Trains the model using the provided dataset.
        """
        num_examples = len(dataset)

        max_num_word_swaps = np.mean([len(x[0]['text'].split(' ')) for x in dataset][:num_examples]) // 20
        if max_num_word_swaps >= 10:
            max_num_word_swaps = 10
        elif max_num_word_swaps <= 1:
            max_num_word_swaps = 1
        else:
            _ = 0

        for attackrecipe in self.attack_recipe:
            if attackrecipe == 'pwws': # word sub
                attack = PWWSRen2019.build(self.model_wrapper)
                attack_class = 'ai'
            # elif attackrecipe == 'pwwsTaip': # add threshold ai as positive
            #     # get threshold
            #     with open(f"{args.output_dir}/predict_results.json", "r") as fin:
            #         metrics = json.load(fin)
            #     if args.attack_class == "ai":
            #         target_max_score = metrics["eval_aip_threshold_chatgpt"]
            #     elif args.attack_class == "human":
            #         target_max_score = metrics["eval_aip_threshold_human"]
            #     else:
            #         raise ValueError('Unknown attack class %s'%args.attack_class)
            #     attack = PWWSRen2019_threshold.build(model_wrapper, target_max_score=target_max_score)
            # elif attackrecipe == 'pwwsThp': # add threshold human as positive
            #     with open(f"{args.output_dir}/predict_results.json", "r") as fin:
            #         metrics = json.load(fin)
            #     if args.attack_class == "ai":
            #         target_max_score = metrics["eval_hp_threshold_chatgpt"]
            #     elif args.attack_class == "human":
            #         target_max_score = metrics["eval_hp_threshold_human"]
            #     else:
            #         raise ValueError('Unknown attack class %s'%args.attack_class)
            #     attack = PWWSRen2019_threshold.build(model_wrapper, target_max_score=target_max_score)
            # elif attackrecipe == 'pruthi': # char sub delete insert etc
            #     attack = Pruthi2019.build(model_wrapper, max_num_word_swaps=max_num_word_swaps)
            # elif attackrecipe == 'deep-word-bug': # word sub, char sub, word del, word insert etc
            #     attack = DeepWordBugGao2018.build(model_wrapper)
            else:
                raise ValueError('Unknown attack recipe %s'%attackrecipe)
            
            attack_args = textattack.AttackArgs(
            num_examples=num_examples,
            log_to_csv='%s/attack_results_%s_%s.csv'%(self.log_path, attack_class, attackrecipe),
            csv_coloring_style='html', 
            )
            attacker = Attacker(attack, dataset, attack_args)
            results = attacker.attack_dataset()
            attacker.attack_log_manager.add_output_file(filename="%s/attack_summary_%s_%s.txt"%(self.log_path, attack_class, attackrecipe), color_method="file")
            attacker.attack_log_manager.log_summary()

