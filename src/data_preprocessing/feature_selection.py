import pandas as pd

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
df_clean = pd.get_dummies(df_clean, columns=['Tipo de Discapacidad'])
df_clean = pd.get_dummies(df_clean, columns=['Condición de Discapacidad'])
df_clean = pd.get_dummies(df_clean, columns=['Pertenencia Étnica'])
df_clean = pd.get_dummies(df_clean, columns=['Coomorbilidad'])
df_clean = pd.get_dummies(df_clean, columns=['OTROS FARMACOS ANTIHIPERTENSIVOS'])
df_clean = pd.get_dummies(df_clean, columns=['OTROS ANTIDIABETICOS'])
df_clean = pd.get_dummies(df_clean, columns=['OTROS TRATAMIENTOS'])
df_clean = pd.get_dummies(df_clean, columns=['OBESIDAD'])

print(df_clean)