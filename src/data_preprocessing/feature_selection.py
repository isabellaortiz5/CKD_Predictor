import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

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

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

# check the shapes of the resulting datasets
print("Training set shape:", X_train_std.shape)
print("Validation set shape:", X_val_std.shape)
print("Testing set shape:", X_test_std.shape)

ckd_df.to_csv('../../data/processed/transformed_data/transformed_data.csv')

mi_scores = mutual_info_classif(X_train, y_train)
print(mi_scores)

selector = SelectKBest(mutual_info_classif, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

selected_columns = selector.get_support(indices=True)
reduced_data_train = pd.DataFrame(X_train_new, columns=ckd_df.columns[selected_columns])
reduced_data_test = pd.DataFrame(X_test_new, columns=ckd_df.columns[selected_columns])


print(X_train_new)
reduced_data_train.to_csv('../../data/processed/transformed_data/reduced_train_transformed_data.csv')
reduced_data_test.to_csv('../../data/processed/transformed_data/reduced_test_transformed_data.csv')