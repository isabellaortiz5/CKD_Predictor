import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import missingno as msno

"""
 "********************************************heatmap of missingness********************************************"
    msno.matrix(df_clean);
    plt.title("heatmap of missingness")
    plt.show()
    
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
    
    '../../data/processed/cleaned_data/Cleaned_data.csv'
"""


class Transform:
    def __init__(self, df_clean_path, transformed_data_path):
        self.df_clean_path = df_clean_path
        self.transformed_data_path = transformed_data_path
        self.df_after_categ_normalization = None
        self.df_after_dummy = None
        self.df_after_data_type = None
        self.df_after_scaling = None
        self.df_transform = None
        self.df_transformed = None

    def load_clean_data(self):
        df_clean = pd.read_csv(self.df_clean_path)
        self.df_transform = df_clean.drop(["Unnamed: 0"], axis=1)

    def categorical_data_normalization(self):
        self.df_transform = self.df_transform.replace('no aplica', 0)
        self.df_transform.loc[self.df_transform['FechaNovedadFallecido'] != 0, 'FechaNovedadFallecido'] = 1
        self.df_transform['ADHERENCIA AL TRATAMIENTO'] = self.df_transform['ADHERENCIA AL TRATAMIENTO'].replace('NO', 0)
        self.df_transform['ADHERENCIA AL TRATAMIENTO'] = self.df_transform['ADHERENCIA AL TRATAMIENTO'].replace('SI', 1)

        self.df_after_categ_normalization = self.df_transform

    def dummifying(self):
        self.df_transform = pd.get_dummies(self.df_transform, columns=['CodDepto'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Tipo de Discapacidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Condición de Discapacidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Pertenencia Étnica'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Coomorbilidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS FARMACOS ANTIHIPERTENSIVOS'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS ANTIDIABETICOS'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS TRATAMIENTOS'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OBESIDAD'])

        self.df_after_dummy = self.df_transform

    def changing_data_type(self):

        self.df_after_data_type = self.df_transform

    def scaling(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        """
        self.df_transform = scaler.fit_transform(self.df_transform)
        self.df_after_scaling = self.df_transform
        """
        
        

    def save(self):
        self.df_transformed = self.df_transform
        self.df_transform = self.df_transform.reset_index(drop=True)
        self.df_transform.to_csv(self.transformed_data_path)
        print("Transformed data succesfully saved in: {}".format(self.transformed_data_path))

    def run(self):
        print("------------------------------------------------")
        print("Transforming...")
        self.load_clean_data()
        self.categorical_data_normalization()
        self.dummifying()
        self.changing_data_type()
        self.scaling()
        print("All transformations successfully applied!")
        self.save()
        print("------------------------------------------------")

    def get_df_transformed(self):
        return self.df_transformed