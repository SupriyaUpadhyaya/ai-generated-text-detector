import numpy as np
from textattack.models.wrappers import SklearnModelWrapper
from src.xgboost_detector.featureExtractor import FeatureExtractor

class XGBoostWrapper(SklearnModelWrapper):
    def __init__(self, model):
        self.model = model
        self.featureExtractor = FeatureExtractor()
        print("Hello model!")
        
    
    def __call__(self, text_input_list):
        print("text_input_list: ", text_input_list)
        input_array = np.array([self.featureExtractor.getFeatures(text) for text in text_input_list])
        print("Input features : ", input_array[0])
        probs = self.model.predict_proba(input_array[0])
        return probs
