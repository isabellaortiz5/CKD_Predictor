import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class MyPCA:
    def __init__(self, df):
        self.df = df
        self.pca = PCA()
        self.df_proyectado = self.pca.fit_transform(df)
        self.var_exp = self.pca.explained_variance_ratio_
        self.cum_var_exp = np.cumsum(self.var_exp)
        self.dataPca = self.pca.transform(df)
        
    def plot_var_exp(self):
        fig = plt.figure(figsize=(15, 7))
        plt.bar(range(len(self.var_exp)), self.var_exp, alpha=0.3333, align='center', label='Varianza explicada por cada PC', color = 'g')
        plt.step(range(len(self.cum_var_exp)), self.cum_var_exp, where='mid',label='Varianza explicada acumulada')
        plt.ylabel('Porcentaje de varianza explicada')
        plt.xlabel('Componentes principales')
        plt.legend(loc='best')
        plt.close()
        return fig
        
    def get_(self):
        return self.pca.components_
    
    def get_components(self):
        return self.pca.components_
    
    def get_columns(self):
        return self.df.columns
    
    def get_explained_variance(self):
        return self.pca.explained_variance_
