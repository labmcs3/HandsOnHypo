import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.model_selection       import train_test_split

df = pd.read_csv("HiggsTraining.csv.zip")
print(df.head())

# Dividiamo il dataset in due parti: X solo features, y solo label
# Droppando le features non interessanti o le label

#Divido in campione di training e test con l'aiuto di train_test_split
#Nella divisione train/test partire da una frazine di train piccola (per testare) per poi aumentarla

#Training di MLP, LD, NaiveBayes
for model in listmodel:
    
    nome = model.__class__.__name__
    # Chiamare fit

    # Calcolare la predizione sul campione di test

    # Fare curva ROC

    # Calcolare AUC

    # Se il modello e' MLP (ad esempio) salviamo il modello in un file per poterlo rileggere
    if nome=="MLPClassifier":
        joblib.dump(model, "MLP_weights.pkl")

plt.show()

