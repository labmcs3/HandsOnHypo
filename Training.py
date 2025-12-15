import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import joblib

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.model_selection       import train_test_split
from sklearn.preprocessing         import StandardScaler

df = pd.read_csv("training.csv.zip")
print(df.head())

# Dividiamo il dataset in due parti: X solo features, y solo label
# Droppando le features non interessanti o le label

# Divido in campione di training e test con l'aiuto di train_test_split
# Nella divisione train/test partire da una frazine di train piccola (per testare) per poi aumentarla

# Eventualmente scalo a norm(0,1) usando lo scaler

#Training di MLP, LD, NaiveBayes
listmodel = []
for model in listmodel:
    
    # Chiamare fit

    # Calcolare la predizione sul campione di test

    # Fare curva ROC

    # Calcolare AUC

    # Se il modello e' MLP (ad esempio) salviamo il modello in un file per poterlo rileggere
    if str(model)=="MLPClassifier()":
        joblib.dump(model, "MLP_weights.pkl")

plt.show()

