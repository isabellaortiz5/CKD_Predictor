import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        self.fe = feature_engineering.feature_eng(self.df_transform)

    def data_types(self):
        self.df_transform = self.df_transform.replace({'nan': np.nan})
        self.df_transform['Grupo de Riesgo'].astype('object').dtypes
        self.df_transform['CodDepto'].astype('float64').dtypes
        self.df_transform['FechaNovedadFallecido'].astype('object').dtypes
        self.df_transform['Edad'].astype('float64').dtypes
        self.df_transform['Cod_Género'].astype('float64').dtypes
        self.df_transform['Tipo de Discapacidad'].astype('object').dtypes
        self.df_transform['Condición de Discapacidad'].astype('object').dtypes
        self.df_transform['Pertenencia Étnica'].astype('object').dtypes
        self.df_transform['Coomorbilidad'].astype('object').dtypes
        self.df_transform['ADHERENCIA AL TRATAMIENTO'].astype('object').dtypes
        self.df_transform['Fumador Activo'].astype('float64').dtypes
        self.df_transform['CONSUMO DE ALCOHOL'].astype('object').dtypes
        self.df_transform['ENTREGA DE MEDICAMENTO OPORTUNA'].astype('object').dtypes
        self.df_transform['FARMACOS ANTIHIPERTENSIVOS'].astype('object').dtypes
        self.df_transform['OTROS FARMACOS ANTIHIPERTENSIVOS'].astype('object').dtypes
        self.df_transform['RECIBE IECA'].astype('object').dtypes
        self.df_transform['RECIBE ARA II'].astype('object').dtypes
        self.df_transform['ESTATINA'].astype('object').dtypes
        self.df_transform['ANTIDIABETICOS'].astype('object').dtypes
        self.df_transform['OTROS ANTIDIABETICOS'].astype('object').dtypes
        self.df_transform['OTROS TRATAMIENTOS'].astype('object').dtypes
        self.df_transform['OTROS DIAGNÓSTICOS'].astype('object').dtypes
        self.df_transform['PESO'].astype('float64').dtypes
        self.df_transform['TALLA'].astype('float64').dtypes
        self.df_transform['IMC'].astype('float64').dtypes
        self.df_transform['OBESIDAD'].astype('object').dtypes
        self.df_transform['CALCULO DE RIESGO DE Framingham (% a 10 años)'].astype('float64').dtypes
        self.df_transform['Clasificación de RCV Global'].astype('object').dtypes
        self.df_transform['DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL'].astype('float64').dtypes
        self.df_transform['CÓD_DIABETES'].astype('float64').dtypes
        self.df_transform['CLASIFICACION DIABETES'].astype('object').dtypes
        self.df_transform['DIAGNÓSTICO DISLIPIDEMIAS'].astype('object').dtypes
        self.df_transform['CÓD_ANTEDECENTE'].astype('float64').dtypes
        self.df_transform['PRESION ARTERIAL'].astype('object').dtypes
        self.df_transform['COLESTEROL ALTO'].astype('object').dtypes
        self.df_transform['HDL ALTO'].astype('object').dtypes
        self.df_transform['CLASIFICACIÓN DE RIESGO CARDIOVASCULAR'].astype('object').dtypes
        self.df_transform['CALCULO TFG'].astype('float64').dtypes
        self.df_transform['CLASIFICACIÓN ESTADIO'].astype('object').dtypes
        self.df_transform['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC'].astype('float64').dtypes
        self.df_transform['GLICEMIA 100 MG/DL_DIC'].astype('float64').dtypes
        self.df_transform['COLESTEROL TOTAL > 200 MG/DL_DIC'].astype('float64').dtypes
        self.df_transform['LDL > 130 MG/DL_DIC'].astype('float64').dtypes
        self.df_transform['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'].astype('float64').dtypes
        self.df_transform['TGD > 150 MG/DL_DIC'].astype('float64').dtypes
        self.df_transform['ALBUMINURIA/CREATINURIA'].astype('float64').dtypes
        self.df_transform['HEMOGLOBINA GLICOSILADA > DE 7%'].astype('float64').dtypes
        self.df_transform['HEMOGRAMA'].astype('object').dtypes
        self.df_transform['POTASIO'].astype('float64').dtypes
        self.df_transform['MICROALBINURIA'].astype('float64').dtypes
        self.df_transform['CREATINURIA'].astype('float64').dtypes
        self.df_transform['UROANALIS'].astype('object').dtypes
        self.df_transform['PERIMETRO ABDOMINAL'].astype('float64').dtypes
        self.df_transform['Complicación Cardiaca'].astype('object').dtypes
        self.df_transform['Complicación Cerebral'].astype('object').dtypes
        self.df_transform['Complicación Retinianas'].astype('object').dtypes
        self.df_transform['Complicación Vascular'].astype('object').dtypes
        self.df_transform['Complicación Renales'].astype('object').dtypes

    def feature_eng(self,df):
        df = self.fe.run()

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

    def dummifying(self):
        self.df_transform = pd.get_dummies(self.df_transform, columns=['CodDepto'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Tipo de Discapacidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Condición de Discapacidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['Pertenencia Étnica'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['Coomorbilidad'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS FARMACOS ANTIHIPERTENSIVOS'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS ANTIDIABETICOS'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['OTROS TRATAMIENTOS'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['OBESIDAD'])
        self.df_transform = pd.get_dummies(self.df_transform, columns=['ENFERMEDADES'])

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
        self.data_types()
        self.df_transform.info(verbose=True)
        self.df_transform = self.feature_eng(self.df_transform)
        self.one_hot_encoding()
        self.changing_data_type()
        self.scaling()
        self.splitting()
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