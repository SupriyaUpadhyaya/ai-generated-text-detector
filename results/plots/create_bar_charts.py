import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


xgboost = pd.read_csv("xgboost.csv")
bloomz = pd.read_csv("bloomz.csv")
roberta = pd.read_csv("roberta.csv")

print(xgboost.head())
print(bloomz.head())
print(roberta.head())

# Select rows 'Alice' to 'Charlie' (inclusive) and column 'City'
xgboost_without_preprocessing = xgboost.loc['Abstract - ChatGPT \n (without preprocessing)':'Abstract - Bloomz \n (without preprocessing)', 'City']
print(single_column)
