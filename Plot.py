import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

df = pd.read_csv('training.csv.zip')
print(df.head())

# Vediamo che ci sono colonne non significative, nel caso le eliminiamo (drop)

# Seleziono solo 10000 eventi per fare veloce

#grafico con subplots
fig, axs = plt.subplots(6, int(len(df.columns)/6)+1, figsize=(20,8))
axs = axs.flatten()

for idx, feat in enumerate(df.columns):
    #...
    #...
#    if idx==0:
#        axs[idx].legend()
    axs[idx].set_xlabel(feat)
    
fig.tight_layout()
plt.show()


