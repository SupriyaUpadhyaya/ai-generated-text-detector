import numpy as np
from textattack.models.wrappers import SklearnModelWrapper
from src.xgboost_detector.featureExtractor import FeatureExtractor

class XGBoostWrapper(SklearnModelWrapper):
    def __init__(self, model):
        self.model = model
        self.featExtractor = FeatureExtractor()
        print("Hello model!")
        
    
    def __call__(self, text_input_list):
        print("text_input_list: ", text_input_list)
        input_feat = self.featExtractor.getFeaturesForAttack(text_input_list)
        #print("Input features : ", input_array[0])
        probs = self.model.predict(input_feat)
        return probs
