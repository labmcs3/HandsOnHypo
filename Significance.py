import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import joblib

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.model_selection       import train_test_split

df = pd.read_csv("HiggsSignificance.csv.zip")
print(df.head())

