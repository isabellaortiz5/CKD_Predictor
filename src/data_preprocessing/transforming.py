import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import missingno as msno

df_clean_path = '../../data/processed/cleaned_data/Cleaned_data.csv'
df_clean = pd.read_csv(df_clean_path)
df_clean = df_clean.drop(["Unnamed: 0"],axis=1)
msno.matrix(df_clean);
plt.title("heatmap of missingness")
plt.show()
"""
Categorical Data normalization
"""
"dummies"
df_clean = pd.get_dummies(df_clean, columns=['Evento'])
df_clean = pd.get_dummies(df_clean, columns=['CodDepto'])
df_clean = pd.get_dummies(df_clean, columns=['Programa'])
df_clean = pd.get_dummies(df_clean, columns=['Tipo de Discapacidad'])
df_clean = pd.get_dummies(df_clean, columns=['Condición de Discapacidad'])
df_clean = pd.get_dummies(df_clean, columns=['Pertenencia Étnica'])
df_clean = pd.get_dummies(df_clean, columns=['Coomorbilidad'])
df_clean = pd.get_dummies(df_clean, columns=['OTROS FARMACOS ANTIHIPERTENSIVOS'])
df_clean = pd.get_dummies(df_clean, columns=['OTROS ANTIDIABETICOS'])
df_clean = pd.get_dummies(df_clean, columns=['OTROS TRATAMIENTOS'])
df_clean = pd.get_dummies(df_clean, columns=['OBESIDAD'])


df_clean['FechaNovedadFallecido'] = df_clean['FechaNovedadFallecido'].replace('no aplica', 0)
df_clean['FechaNovedadFallecido'][df_clean['FechaNovedadFallecido']!=0] = 1
df_clean['ADHERENCIA AL TRATAMIENTO'] = df_clean['ADHERENCIA AL TRATAMIENTO'].replace('NO', 0)
df_clean['ADHERENCIA AL TRATAMIENTO'] = df_clean['ADHERENCIA AL TRATAMIENTO'].replace('SI', 1)

print("****************************** Changing data type ******************************")
#df_clean = df_clean.astype('float64')

"""
Scaling to a range
"""

scaler = MinMaxScaler(feature_range=(0, 1))
df_clean['Edad'] = scaler.fit_transform(df_clean[['Edad']])
df_clean['CiclosV'] = scaler.fit_transform(df_clean[['CiclosV']])
#df_clean['IMC'] = scaler.fit_transform(df_clean[['IMC']])
#df_clean['CALCULO DE RIESGO DE Framingham (% a 10 años)'] = scaler.fit_transform(df_clean[['CALCULO DE RIESGO DE Framingham (% a 10 años)']])

print("****************************** Scaling to a range ******************************")
print(df_clean.info())
print(df_clean.head())

print("****************************** Normalization ******************************")

#df_zscore = df_clean.apply(stats.zscore)

# Print the normalized DataFrame
#print(df_zscore)

print("****************************** Correlation matrix ******************************")
def plot_correlation_matrix(df, graph_width):
    df = df.dropna('columns') # drop columns with NaN
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
    plt.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=1)
    plt.show()
    print(corr)


print(df_clean.corr())
plot_correlation_matrix(df_clean, 100)