import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

df_clean_path = '../../data/processed/cleaned_data/Cleaned_data.csv'
df_clean = pd.read_csv(df_clean_path)
df_clean = df_clean.drop(["Unnamed: 0"],axis=1)

print(df_clean.info())
df_clean = df_clean.replace('no aplica', 0)
df_clean['FechaNovedadFallecido'][df_clean['FechaNovedadFallecido']!=0] = 1
df_clean['ADHERENCIA AL TRATAMIENTO'] = df_clean['ADHERENCIA AL TRATAMIENTO'].replace('NO', 0)
df_clean['ADHERENCIA AL TRATAMIENTO'] = df_clean['ADHERENCIA AL TRATAMIENTO'].replace('SI', 1)

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
df_clean = pd.get_dummies(df_clean, columns=['ADHERENCIA AL TRATAMIENTO'])

ckd_df = df_clean

X = ckd_df.drop('Evento_ERC', axis=1)
y = ckd_df['Evento_ERC']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# check the shapes of the resulting datasets
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Testing set shape:", X_test.shape)

df_clean.to_csv('../../data/processed/transformed_data/transformed_data.csv')
scaler = StandardScaler()
X = scaler.fit_transform(X)

mi_scores = mutual_info_classif(X, y)