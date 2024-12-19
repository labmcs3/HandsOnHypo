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

# Vediamo che ci sono colonne non si significative (Weight)
# -> le eliminiamo (drop)


# Seleziono solo 10000 eventi per fare veloce

#grafico con subplots
#fig, axs = plt.subplots(6, 5, figsize=(20,8))
#axs = axs.flatten()

#grafico le distribuzioni con seaborn
#for col,feat in enumerate(df.columns):
#    if feat != "Label":
#        if col!=0:
#            #data -> dataset, x -> nome della variabile, hue -> categoria che determina il colore delle componenti
#            sns.histplot(data=...,x=...,hue=..., ax=axs[col], legend=False, alpha=0.6, stat="density",common_norm=False)
#        else:
#            sns.histplot(data=...,x=...,hue=..., ax=axs[col], legend=True, alpha=0.6, stat="density",common_norm=False)
#fig.tight_layout()
#plt.show()


