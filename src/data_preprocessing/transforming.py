import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_clean_path = '../../data/processed/cleaned_data/Cleaned_data.csv'
df_clean = pd.read_csv(df_clean_path)
df_clean = df_clean.drop(["Unnamed: 0"],axis=1)

"""
Categorical Data normalization
"""
"dummies"
df_clean = pd.get_dummies(df_clean, columns=['Evento'])
df_clean = pd.get_dummies(df_clean, columns=['CodDepto'])
df_clean = pd.get_dummies(df_clean, columns=['Programa'])

"fallecido"
df_clean = df_clean.replace()

print("****************************** Categorical Data normalization ******************************")
print(df_clean.info())

"""
Scaling to a range
"""

scaler = MinMaxScaler(feature_range=(0, 1))
df_clean['Edad'] = scaler.fit_transform(df_clean[['Edad']])

print("****************************** Scaling to a range ******************************")
print(df_clean['Edad'].head())
print(df_clean.info())


def plot_correlation_matrix(df, graph_width):
    # df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graph_width, graph_width), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for ', fontsize=15)
    plt.show()


print(df_clean.corr())
plot_correlation_matrix(df_clean, 10)