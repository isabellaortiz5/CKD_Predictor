import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# check the shapes of the resulting datasets
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Testing set shape:", X_test.shape)


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

"""
PCA
"""

# Fit PCA on the training data
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_std)


# Transform the validation and test data
X_val_pca = pca.transform(X_val_std)
X_test_pca = pca.transform(X_test_std)


# Train a logistic regression model on the transformed training data
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)

# Evaluate the model on the transformed validation and test data
y_val_pred = lr.predict(X_val_pca)
val_acc = accuracy_score(y_val, y_val_pred)

y_test_pred = lr.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_test_pred)

print("Accuracy PCA:", test_acc)

# Plot explained variance ratio of each component
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA variance ratio')
plt.show()

# Convert the transformed data to a Pandas DataFrame
X_train_pca_df = pd.DataFrame(X_train_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10'])
X_val_pca_df = pd.DataFrame(X_val_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10'])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10'])

# Add original column names to PCA data
X_train_pca_df = pd.concat([X_train_pca_df, pd.DataFrame(X_train, columns=X.columns)], axis=1)
X_val_pca_df = pd.concat([X_val_pca_df, pd.DataFrame(X_val, columns=X.columns)], axis=1)
X_test_pca_df = pd.concat([X_test_pca_df, pd.DataFrame(X_test, columns=X.columns)], axis=1)

X_train_pca_df.to_csv('../../data/processed/transformed_data/pca_transformed_data.csv')

"""
LDA
"""


# Fit LDA on the training data
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# Transform the validation and test data
X_val_lda = lda.transform(X_val_std)
X_test_lda = lda.transform(X_test_std)

# Train a logistic regression model on the transformed training data
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)

# Evaluate the model on the transformed validation and test data
y_val_pred = lr.predict(X_val_lda)
val_acc = accuracy_score(y_val, y_val_pred)

y_test_pred = lr.predict(X_test_lda)
test_acc = accuracy_score(y_test, y_test_pred)

print("Accuracy LDA:", test_acc)

# Plot LDA explained variance ratio of each component
plt.plot(lda.explained_variance_ratio_)
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.title('LDA variance ratio')
plt.show()

"""
t-SNE
"""


# Fit t-SNE on the training data
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_train_tsne = tsne.fit_transform(X_train_std)

# Transform the validation and test data
X_val_tsne = tsne.transform(X_val_std)
X_test_tsne = tsne.transform(X_test_std)

# Plot LDA explained variance ratio of each component
plt.plot(tsne.explained_variance_ratio_)
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.title('t-sne variance ratio')
plt.show()


