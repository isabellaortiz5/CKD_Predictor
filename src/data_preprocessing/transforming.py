import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import missingno as msno
import feature_engineering

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

    def data_types(self, df):
        df = df.replace({'nan': np.nan})
        df = df.replace({'NAN': np.nan})
        df = df.astype({'Grupo de Riesgo': 'object',
            'CodDepto': 'float64',
            'FechaNovedadFallecido': 'object',
            'Edad': 'float64',
            'Cod_Género': 'float64',
            'Tipo de Discapacidad': 'object',
            'Condición de Discapacidad': 'object',
            'Pertenencia Étnica': 'object',
            'Coomorbilidad': 'object',
            'ADHERENCIA AL TRATAMIENTO': 'object',
            'Fumador Activo': 'float64',
            'CONSUMO DE ALCOHOL': 'object',
            'ENTREGA DE MEDICAMENTO OPORTUNA': 'object',
            'FARMACOS ANTIHIPERTENSIVOS': 'object',
            'OTROS FARMACOS ANTIHIPERTENSIVOS': 'object',
            'RECIBE IECA': 'object',
            'RECIBE ARA II': 'object',
            'ESTATINA': 'object',
            'ANTIDIABETICOS': 'object',
            'OTROS ANTIDIABETICOS': 'object',
            'OTROS TRATAMIENTOS': 'object',
            'OTROS DIAGNÓSTICOS': 'object',
            'PESO': 'float64',
            'TALLA': 'float64',
            'IMC': 'float64',
            'OBESIDAD': 'object',
            'CALCULO DE RIESGO DE Framingham (% a 10 años)': 'float64',
            'Clasificación de RCV Global': 'object',
            'DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL': 'float64',
            'CÓD_DIABETES': 'float64',
            'CLASIFICACION DIABETES': 'object',
            'DIAGNÓSTICO DISLIPIDEMIAS': 'object',
            'CÓD_ANTEDECENTE': 'float64',
            'PRESION ARTERIAL': 'object',
            'COLESTEROL ALTO': 'object',
            'HDL ALTO': 'object',
            'CLASIFICACIÓN DE RIESGO CARDIOVASCULAR': 'object',
            'CALCULO TFG': 'float64',
            'CLASIFICACIÓN ESTADIO': 'object',
            'CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC': 'float64',
            'GLICEMIA 100 MG/DL_DIC': 'float64',
            'COLESTEROL TOTAL > 200 MG/DL_DIC': 'float64',
            'LDL > 130 MG/DL_DIC': 'float64',
            'HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC': 'float64',
            'TGD > 150 MG/DL_DIC': 'float64',
            'ALBUMINURIA/CREATINURIA': 'float64',
            'HEMOGLOBINA GLICOSILADA > DE 7%': 'float64',
            'HEMOGRAMA': 'object',
            'POTASIO': 'float64',
            'MICROALBINURIA': 'float64',
            'CREATINURIA': 'float64',
            'UROANALIS': 'object',
            'PERIMETRO ABDOMINAL': 'float64',
            'Complicación Cardiaca': 'object',
            'Complicación Cerebral': 'object',
            'Complicación Retinianas': 'object',
            'Complicación Vascular': 'object',
            'Complicación Renales': 'object'})

        return df

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
            float_dict = {}
            for i, val in enumerate(unique_values):
                float_dict[val] = float(i)
            self.df_transform[col] = self.df_transform[col].map(float_dict)
            print(f"Integer encoding for column '{col}': {float_dict}")

    def mean_calculator(self, df):
        return df

    def mice_imputation(self, df):
        df_mice = df.filter(['CALCULO DE RIESGO DE Framingham (% a 10 años)',
                                  'GLICEMIA 100 MG/DL_DIC',
                                  'COLESTEROL TOTAL > 200 MG/DL_DIC',
                                  'CALCULO TFG',
                                  'CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC',
                                  'LDL > 130 MG/DL_DIC','HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC',
                                  'TGD > 150 MG/DL_DIC',
                                  'HEMOGLOBINA GLICOSILADA > DE 7%',
                                  'MICROALBINURIA','CREATINURIA'
                                  ], axis=1).copy()
        
        n_imputations = 5
        imputed_datasets = []

        for i in range(n_imputations):
            # Initialize the Iterative Imputer with a different random state for each imputation
            iterative_imputer = IterativeImputer(max_iter=10, random_state=i)
            
            # Impute missing values
            data_imputed = iterative_imputer.fit_transform(data)
            
            # Convert the imputed data back to a pandas DataFrame with the original column names
            data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)
            
            # Append the imputed DataFrame to the list of imputed datasets
            imputed_datasets.append(data_imputed_df)


        # Define MICE Imputer and fill missing values
        mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None,
                                        imputation_order='ascending')

        df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)

        df['CALCULO DE RIESGO DE Framingham (% a 10 años)'] = df_mice_imputed['CALCULO DE RIESGO DE Framingham (% a 10 años)']
        df['GLICEMIA 100 MG/DL_DIC'] = df_mice_imputed['GLICEMIA 100 MG/DL_DIC']
        df['COLESTEROL TOTAL > 200 MG/DL_DIC'] = df_mice_imputed['COLESTEROL TOTAL > 200 MG/DL_DIC']
        df['CALCULO TFG'] = df_mice_imputed['CALCULO TFG']
        df['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC'] = df_mice_imputed['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC']
        df['LDL > 130 MG/DL_DIC'] = df_mice_imputed['LDL > 130 MG/DL_DIC']
        df['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'] = df_mice_imputed['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC']
        df['TGD > 150 MG/DL_DIC'] = df_mice_imputed['TGD > 150 MG/DL_DIC']
        df['HEMOGLOBINA GLICOSILADA > DE 7%'] = df_mice_imputed['HEMOGLOBINA GLICOSILADA > DE 7%']
        df['MICROALBINURIA'] = df_mice_imputed['MICROALBINURIA']
        df['CREATINURIA'] = df_mice_imputed['CREATINURIA']


        df['ALBUMINURIA/CREATINURIA'] = df_mice_imputed['ALBUMINURIA/CREATINURIA']
        return df
    
    def calculate_erc_stage(self, df):
        df["Calculo_ERC"] = np.nan

        for i,row in df.iterrows():
            if(not (row['CALCULO TFG'] is np.nan )):
                if( row['CALCULO TFG'] > 90):
                    df['Calculo_ERC'][i] = 1

                else:
                    if( row['CALCULO TFG'] > 59):
                        df['Calculo_ERC'][i] = 2

                    else:
                        if( row['CALCULO TFG'] > 44 ):
                            df['Calculo_ERC'][i] = 3.1

                        else:
                            if( row['CALCULO TFG'] > 29):
                                df['Calculo_ERC'][i] = 3.2

                            else:
                                if( row['CALCULO TFG'] > 15):
                                    df['Calculo_ERC'][i] = 4

                                else:
                                    df['Calculo_ERC'][i] = 5

            else:
                df['Calculo_ERC'][i] = np.nan 
        
        df.drop('CLASIFICACIÓN ESTADIO', axis=1)

        return df
    
    def calculate_erc_stage_albuminuria(self, df):
        df["Calculo_ERC_ALBUMINURIA"] = np.nan

        for i,row in df.iterrows():
            if(not (row['ALBUMINURIA/CREATINURIA'] is np.nan )):
                if( row['ALBUMINURIA/CREATINURIA'] > 90):
                    df['Calculo_ERC'][i] = 1

                else:
                    if( row['CALCULO TFG'] > 59):
                        df['Calculo_ERC'][i] = 2

                    else:
                        if( row['CALCULO TFG'] > 44 ):
                            df['Calculo_ERC'][i] = 3.1

                        else:
                            if( row['CALCULO TFG'] > 29):
                                df['Calculo_ERC'][i] = 3.2

                            else:
                                if( row['CALCULO TFG'] > 15):
                                    df['Calculo_ERC'][i] = 4

                                else:
                                    df['Calculo_ERC'][i] = 5

            else:
                df['Calculo_ERC'][i] = np.nan 
        
        df.drop('CLASIFICACIÓN ESTADIO', axis=1)

        return df

    def dummifying(self):
        self.df_transform = pd.get_dummies(self.df_transform, columns=['CodDepto'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Tipo de Discapacidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Condición de Discapacidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Pertenencia Étnica'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['Coomorbilidad'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS FARMACOS ANTIHIPERTENSIVOS'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS ANTIDIABETICOS'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS TRATAMIENTOS'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OBESIDAD'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['ENFERMEDADES'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['FARMACOS'])

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
        X = ckd_df.drop('Calculo_ERC', axis=1)
        y = ckd_df['Calculo_ERC']
        
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

    @staticmethod
    def get_nan_per_col(df):
        nan_percentage = ((df.isna().sum() * 100) / df.shape[0]).sort_values(ascending=True)
        return nan_percentage
    
    @staticmethod
    def remove_by_nan(accepted_nan_percentage, nan_percentage_series, df):
        columns_dropped = []
        for items in nan_percentage_series.items():
            if items[1] > accepted_nan_percentage:
                df = df.drop([items[0]], axis=1)
                columns_dropped.append(items)

        return df, columns_dropped

    def drop_nan(self, df):
        nan_percentages = self.get_nan_per_col(df)
        df, columns_dropped = self.remove_by_nan(46, nan_percentages, df)
        df = df.dropna()
        print('COLUMS DROPPED: ', columns_dropped)
        return df
        


    def run(self):
        print("------------------------------------------------")
        print("Transforming...")
        self.load_clean_data()
        self.df_transform = self.data_types(self.df_transform)
        msno.matrix(self.df_transform, sparkline=False)
        self.df_transform = self.mice_imputation(self.df_transform)
        self.df_transform = self.fe.run(self.df_transform)
        msno.matrix(self.df_transform, sparkline=False)
        self.df_transform = self.drop_nan(self.df_transform)
        msno.matrix(self.df_transform, sparkline=False)
        
        self.df_transform = self.calculate_erc_stage(self.df_transform)
        
        self.one_hot_encoding()
        self.changing_data_type()
        self.scaling()
        self.splitting()
        #TODO: cast all df to float
        msno.matrix(self.df_transform, sparkline=False)
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