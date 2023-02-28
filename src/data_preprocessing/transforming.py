
"""
Categorical Data normalization
"""
one_hot_data = pd.get_dummies(df_clean, columns=['Evento'])
df_clean = pd.get_dummies(one_hot_data, columns=['CodDepto'])
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

def plotCorrelationMatrix(df, graphWidth):
  # df = df.dropna('columns') # drop columns with NaN
  df = df[[col for col in df if df[col].nunique() > 1]]
  if df.shape[1] < 2:
    print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
    return
  corr = df.corr()
  plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
  corrMat = plt.matshow(corr, fignum=1)
  plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
  plt.yticks(range(len(corr.columns)), corr.columns)
  plt.gca().xaxis.tick_bottom()
  plt.colorbar(corrMat)
  plt.title(f'Correlation Matrix for ', fontsize=15)
  plt.show()

print(df_clean.corr())
plotCorrelationMatrix(df_clean, 10)