import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import missingno as msno
import feature_engineering
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
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_val = None
        self.y_test = None
        self.y_train = None
        self.df_after_scaling = None
        self.df_transform = None
        self.df_transformed = None
        self.fe = None

    def load_clean_data(self):
        df_clean = pd.read_csv(self.df_clean_path)
        self.df_transform = df_clean.drop(["Unnamed: 0"], axis=1)
        self.fe = feature_engineering.feature_eng()

    
    def general_categorical_data(self):
        self.df_transform = self.df_transform.replace('no aplica', 0)
        self.df_transform.loc[self.df_transform['FechaNovedadFallecido'] != 0, 'FechaNovedadFallecido'] = 1
        self.df_transform['ADHERENCIA AL TRATAMIENTO'] = self.df_transform['ADHERENCIA AL TRATAMIENTO'].replace('NO', 0)
        self.df_transform['ADHERENCIA AL TRATAMIENTO'] = self.df_transform['ADHERENCIA AL TRATAMIENTO'].replace('SI', 1)

        self.df_after_categ_normalization = self.df_transform
    
    def category_encoding(self):
        obj_cols = list(self.df_transform.select_dtypes(include=['object']).columns)
        for col in obj_cols:
            unique_values = self.df_transform[col].unique()
            int_dict = {}
            for i, val in enumerate(unique_values):
                int_dict[val] = i
            self.df_transform[col] = self.df_transform[col].map(int_dict)
            print(f"Integer encoding for column '{col}': {int_dict}")

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

    def one_hot_encoding(self):
        self.general_categorical_data()
        self.dummifying()
        self.category_encoding()

    def changing_data_type(self):
        self.df_after_data_type = self.df_transform

    def scaling(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        """
        self.df_transform = scaler.fit_transform(self.df_transform)
        self.df_after_scaling = self.df_transform
        
         scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        """

    def splitting(self):
        ckd_df = self.df_transform
        X = ckd_df.drop('Clasificación de RCV Global', axis=1)
        y = ckd_df['Clasificación de RCV Global']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_val = y_val
        self.y_test = y_test
        self.y_train = y_train

    def save(self):
        self.df_transformed = self.df_transform
        self.df_transform = self.df_transform.reset_index(drop=True)

        self.df_transform.to_csv(str("{}/transformed_data.csv".format(self.transformed_data_path)))
        self.X_train.to_csv(str("{}/X_train.csv".format(self.transformed_data_path)))
        self.X_val.to_csv(str("{}/X_val.csv".format(self.transformed_data_path)))
        self.X_test.to_csv(str("{}/X_test.csv".format(self.transformed_data_path)))
        self.y_val.to_csv(str("{}/y_val.csv".format(self.transformed_data_path)))
        self.y_test.to_csv(str("{}/y_test.csv".format(self.transformed_data_path)))
        self.y_train.to_csv(str("{}/y_train.csv".format(self.transformed_data_path)))

        print("Transformed data(Before split) succesfully saved in: {}".format(self.transformed_data_path))
        print("X_train succesfully saved in: {}".format(self.transformed_data_path))
        print("X_val succesfully saved in: {}".format(self.transformed_data_path))
        print("X_test succesfully saved in: {}".format(self.transformed_data_path))
        print("y_val succesfully saved in: {}".format(self.transformed_data_path))
        print("y_test succesfully saved in: {}".format(self.transformed_data_path))
        print("y_train succesfully saved in: {}".format(self.transformed_data_path))

    def run(self):
        print("------------------------------------------------")
        print("Transforming...")
        self.load_clean_data()
        self.df_transform = self.fe.run()
        self.one_hot_encoding()
        self.changing_data_type()
        self.scaling()
        self.splitting()
        print("All transformations successfully applied!")
        self.save()
        print("------------------------------------------------")

    def get_df_transformed(self):
        return self.df_transformed
    
    def get_x_train_transformed(self):
        return self.X_train
    
    def get_x_val_transformed(self):
        return self.X_val
    
    def get_x_test_transformed(self):
        return self.X_test
    
    def get_y_val_transformed(self):
        return self.y_val
    
    def get_y_test_transformed(self):
        return self.y_test
    
    def get_y_train_transformed(self):
        return self.y_train