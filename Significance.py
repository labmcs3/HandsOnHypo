import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import joblib

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.model_selection       import train_test_split

df = pd.read_csv("test.csv.zip")
print(df.head())

#Carico il modello
model = joblib.load("MLP_weights.pkl")

Test = df.drop(columns=['Weight','Label'])

vscore = model.predict_proba(Test)[:,1]
weight = df['Weight']
label  = df['Label']

#Calcolare il numero di eventi (somma dei pesi) che passano il taglio per il campione di segnale e di fondo

#Challenge: trovare il valore di cut che massimizza S/sqrt(B)
#A che valore di efficienza corrisponde ?
