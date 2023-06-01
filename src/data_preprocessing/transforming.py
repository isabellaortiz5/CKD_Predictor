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
        df = df.dropna()
        mean_list = []

        mean_list.append(df['CALCULO DE RIESGO DE Framingham (% a 10 años)'].mean())
        mean_list.append(df['GLICEMIA 100 MG/DL_DIC'].mean())
        mean_list.append(df['COLESTEROL TOTAL > 200 MG/DL_DIC'].mean())
        mean_list.append(df['CALCULO TFG'].mean())
        mean_list.append(df['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC'].mean())
        mean_list.append(df['LDL > 130 MG/DL_DIC'].mean())
        mean_list.append(df['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'].mean())
        mean_list.append(df['TGD > 150 MG/DL_DIC'].mean())
        mean_list.append(df['HEMOGLOBINA GLICOSILADA > DE 7%'].mean())
        mean_list.append(df['CREATINURIA'].mean())
        mean_list.append(df['MICROALBINURIA'].mean())

        return mean_list

    def get_best_dataset(self, base_df, imputed_datasets):
        target_cols = ['CALCULO DE RIESGO DE Framingham (% a 10 años)',
                                  'GLICEMIA 100 MG/DL_DIC',
                                  'COLESTEROL TOTAL > 200 MG/DL_DIC',
                                  'CALCULO TFG',
                                  'CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC',
                                  'LDL > 130 MG/DL_DIC',
                                  'HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC',
                                  'TGD > 150 MG/DL_DIC',
                                  'HEMOGLOBINA GLICOSILADA > DE 7%',
                                  'MICROALBINURIA','CREATINURIA','ALBUMINURIA/CREATINURIA'
                                  ]
        
        #asign the fisrst imputed df as the best 
        Best_columns_df = imputed_datasets[0].filter(target_cols, axis=1).copy()
        
        for df_inputed in imputed_datasets:
            for target in target_cols:
                # calculate mean values for each dataframe for the target col
                df_mean = df_inputed[target].mean()
                base_df_mean = base_df[target].mean()
                best_columns_df_mean = Best_columns_df[target].mean()

                # find which mean value is closest to base_df_mean
                diff1 = abs(df_mean - base_df_mean)
                diff2 = abs(best_columns_df_mean - base_df_mean)
                min_diff = np.min([diff1, diff2])

                if min_diff == diff1:
                    #df_inputed is closer to base_df so its asignet to the Best_columns_df
                    Best_columns_df[target] = df_inputed[target]



        return Best_columns_df

    def mice_imputation(self, df):
        before_inputation_means = self.mean_calculator(df)

        data = df.filter(['CALCULO DE RIESGO DE Framingham (% a 10 años)',
                                  'GLICEMIA 100 MG/DL_DIC',
                                  'COLESTEROL TOTAL > 200 MG/DL_DIC',
                                  'CALCULO TFG',
                                  'CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC',
                                  'LDL > 130 MG/DL_DIC',
                                  'HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC',
                                  'TGD > 150 MG/DL_DIC',
                                  'HEMOGLOBINA GLICOSILADA > DE 7%',
                                  'MICROALBINURIA','CREATINURIA','ALBUMINURIA/CREATINURIA'
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

        df_mice_imputed = self.get_best_dataset(data, imputed_datasets)


        """
        # Define MICE Imputer and fill missing values
        mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None,
                                        imputation_order='ascending')

        df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)
        """
        
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
    
    @staticmethod
    def calculate_erc_stage(df):
        def calculate(row):
            if not pd.isna(row['CALCULO TFG']):
                if row['CALCULO TFG'] > 90:
                    return 1
                elif row['CALCULO TFG'] > 59:
                    return 2
                elif row['CALCULO TFG'] > 44:
                    return 3.1
                elif row['CALCULO TFG'] > 29:
                    return 3.2
                elif row['CALCULO TFG'] > 15:
                    return 4
                else:
                    return 5
            else:
                return np.nan
        df = df.reset_index(drop=True)
        df["Calculo_ERC"] = df.apply(calculate, axis=1)
        return df

    @staticmethod
    def calculate_erc_stage_albuminuria(df):
        def calculate(row):
            if not pd.isna(row['MICROALBINURIA']):
                if row['MICROALBINURIA'] < 30:
                    return 1
                elif row['MICROALBINURIA'] <= 300:
                    return 2 
                else:
                    return 3
            else:
                return np.nan
        df = df.reset_index(drop=True)
        df["Calculo_ERC_ALBUMINURIA"] = df.apply(calculate, axis=1)
        df = df.drop('MICROALBINURIA', axis=1)
        return df
    
    @staticmethod
    def asignar_riesgo(df):
        def calcular_nivel_riesgo(row):
            if row['CALCULO TFG'] >= 90:
                if row['Calculo_ERC_ALBUMINURIA'] < 30:
                    return 'bajo'
                elif row['Calculo_ERC_ALBUMINURIA'] < 300:
                    return 'moderado'
                else:
                    return 'muy alto'
            elif row['CALCULO TFG'] >= 60:
                if row['Calculo_ERC_ALBUMINURIA'] < 30:
                    return 'bajo'
                elif row['Calculo_ERC_ALBUMINURIA'] < 300:
                    return 'moderado'
                else:
                    return 'muy alto'
            elif row['CALCULO TFG'] >= 30:
                if row['Calculo_ERC_ALBUMINURIA'] < 30:
                    return 'moderado'
                elif row['Calculo_ERC_ALBUMINURIA'] < 300:
                    return 'alto'
                else:
                    return 'muy alto'
            elif row['CALCULO TFG'] >= 15:
                if row['Calculo_ERC_ALBUMINURIA'] < 30:
                    return 'moderado'
                elif row['Calculo_ERC_ALBUMINURIA'] < 300:
                    return 'alto'
                else:
                    return 'muy alto'
            else:
                return 'muy alto'

        df['nivel_riesgo'] = df.apply(calcular_nivel_riesgo, axis=1)
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
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['ENFERMEDADES'])
        #self.df_transform = pd.get_dummies(self.df_transform, columns=['FARMACOS'])

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
        X = ckd_df.drop('nivel_riesgo', axis=1)
        y = ckd_df['nivel_riesgo']
        
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
    
    @staticmethod
    def create_comorbidity_columns(df):
        comorbidity_dict = {
            "Certain infectious or parasitic diseases": ["Malaria", "Tuberculosis", "Chagas"],
            "Neoplasms": ["Cáncer de pulmón", "Cáncer de mama", "Leucemia", "CA DE COLON", "LINFOMA HODKING", "CA MAMA"],
            "Diseases of the blood or blood-forming organs": ["Anemia", "Hemofilia", "Leucopenia"],
            "Diseases of the immune system": ["Lupus", "Artritis reumatoide", "Esclerosis múltiple"],
            "Endocrine, nutritional or metabolic diseases": ["Diabetes", "Hipotiroidismo", "Obesidad", "DM INSULINODEPENDIENTE", "PREDIABETES", "HIPERGLICEMIA", "HIPERCOLESTEROLEMIA", "E039", "E669", "E039+N189", "E784", "E780"],
            "Mental, behavioural or neurodevelopmental disorders": ["Esquizofrenia", "Trastorno bipolar", "Autismo", "ANSIEDAD", "DEPRESION"],
            "Sleep-wake disorders": ["Insomnio", "Narcolepsia", "Apnea del sueño", "G473"],
            "Diseases of the nervous system": ["Enfermedad de Parkinson", "Alzheimer", "Epilepsia", "G589", "G819+O694", "G819", "G309", "G510", "G510+G459", "G20X"],
            "Diseases of the visual system": ["Cataratas", "Glaucoma", "Miopía", "H409", "H408", "H360"],
            "Diseases of the ear or mastoid process": ["Otitis", "Tinnitus", "Mastoiditis"],
            "Diseases of the circulatory system": ["Hipertensión arterial", "Insuficiencia cardíaca", "Infarto de miocardio", "HTA", "HTA+DM", "HTA+ERC", "I255", "I694, EO39", "I694+G819", "I694"],
            "Diseases of the respiratory system": ["Asma", "EPOC", "Neumonía", "J449", "J449, K219"],
            "Diseases of the digestive system": ["Gastritis", "Enfermedad de Crohn", "Cirrosis", "K219"],
            "Diseases of the skin": ["Acné", "Psoriasis", "Eczema"],
            "Diseases of the musculoskeletal system or connective tissue": ["Artritis", "Osteoporosis", "Fibromialgia", "M719", "M819", "M541", "M069", "M139"],
            "Diseases of the genitourinary system": ["Infección urinaria", "Insuficiencia renal crónica", "Cistitis", "N40X", "N189"],
            "Conditions related to sexual health": ["VIH", "Sífilis", "Gonorrea"],
            "Pregnancy, childbirth or the puerperium": ["Preeclampsia", "Diabetes gestacional", "Depresión posparto", "O694"],
            "Certain conditions originating in the perinatal period": ["Nacimiento prematuro", "Bajo peso al nacer", "Asfixia perinatal"],
            "Developmental anomalies": ["Labio leporino", "Espina bífida", "Síndrome de Down", "Q613+N19X", "Q613"],
            "Symptoms, signs or clinical findings, not elsewhere classified": ["Fatiga", "Dolor de cabeza", "Pérdida de peso", "R51X", "R804"],
            "Injury, poisoning or certain other consequences of external causes": ["Fracturas", "Quemaduras", "Intoxicación por alimentos", "F102, F132"],
            "External causes of morbidity or mortality": ["Accidentes de tráfico", "Caídas", "Ahogamiento"],
            "Factors influencing health status or contact with health services": ["Vacunación", "Chequeo médico", "Rehabilitación", "TRANSPLANTE RENAL"]
        }
        # For each comorbidity in the dictionary, create a new column in the dataframe and set its initial value to 0
        for comorbidity in comorbidity_dict.keys():
            df[comorbidity] = 0
            
            # For each subtype of the comorbidity, set the value of the corresponding column in the dataframe to 1 if the subtype is present in the comorbidity column
            for subtype in comorbidity_dict[comorbidity]:
                df.loc[df['ENFERMEDADES'].str.contains(subtype,  na=False), comorbidity] = 1
        

        df = df.drop(["ENFERMEDADES"], axis=1)

        # Return the modified dataframe
        return df
    
    @staticmethod
    def create_farmacos_columns(df):
        farmacos_dict = {
            "Alimentary tract and metabolism": ["METFORMINA", "VIDALGLIPTINA", "GLIBENCLAMIDA", "SITAGLIPTINA", "LINAGLIPTINA", "GEMFIBROZIL", "ALOPURINOL", "CARVELIDOL", "ASA", "FUROSEMIDA", "LEVOTIROXINA"],
            "Blood and blood forming organs": ["HCTZ", "HIDROCODONA", "NIMODIPINO", "CANDESARTAN"],
            "Cardiovascular system": ["AMLODIPINO", "LOSARTAN", "CARVEDILOL", "VERAPAMILO", "NIFEDIPINO", "DILTIAZEM", "BISOPROLOL"],
            "Dermatologicals": ["FENOFIBRATO", "DIOSMINA", "ESPIRONOLACTONA", "FUROSEMIDA"],
            "Genito urinary system and sex hormones": ["PROPANOLOL", "GALVUSMET", "QUINAPRIL"],
            "Systemic hormonal preparations, excluding sex hormones and insulins": ["LEVOTIROXINA", "INSULINA", "GLARGINA", "METOPROLOL", "ROSUVASTATINA", "ACIDO FENOFIBRICO"],
            "Antiinfective for systemic use": ["AMOXICILINA", "CIPROFLOXACINA", "CLARITROMICINA", "DAPAGLIFLOXINA", "CLONIDINA"],
            "Antineoplastic and immunomodulating agents": ["CISPLATINO", "METOTREXATO", "TRASTUZUMAB"],
            "Musculo-skeletal system": ["IBUPROFENO", "PARACETAMOL", "NAPROXENO"],
            "Nervous system": ["PAROXETINA", "RIVASTIGMINA", "QUETIAPINA"],
            "Antiparasitic products, insecticides and repellents": ["MEBENDAZOL", "PERMETRINA", "DEET"],
            "Respiratory system": ["SALBUTAMOL", "BUDESONIDA", "MONTELUKAST"],
            "Sensory organs": ["TIMOLOL", "BRIMONIDINA", "TOBRAMICINA"],
            "Various": ["DIMETILSULFOXIDO", "MANNITOL", "PROPILGALLATO"]
        }
        # For each comorbidity in the dictionary, create a new column in the dataframe and set its initial value to 0
        for farmaco in farmacos_dict.keys():
            df[farmaco] = 0
            
            # For each subtype of the comorbidity, set the value of the corresponding column in the dataframe to 1 if the subtype is present in the comorbidity column
            for subtype in farmacos_dict[farmaco]:
                df.loc[df['FARMACOS'].str.contains(subtype,  na=False), farmaco] = 1
        

        df = df.drop(["FARMACOS"], axis=1)

        # Return the modified dataframe
        return df

    def run(self):
        print("------------------------------------------------")
        print("Transforming...")
        self.load_clean_data()
        self.df_transform = self.data_types(self.df_transform)
        msno.matrix(self.df_transform, sparkline=False)
        self.df_transform = self.mice_imputation(self.df_transform)
        self.df_transform = self.fe.run(self.df_transform)
        self.df_transform = self.calculate_erc_stage(self.df_transform)
        self.df_transform = self.calculate_erc_stage_albuminuria(self.df_transform)
        self.df_transform = self.asignar_riesgo(self.df_transform)
        self.df_transform = self.create_comorbidity_columns(self.df_transform)
        self.df_transform = self.create_farmacos_columns(self.df_transform)

        msno.matrix(self.df_transform, sparkline=False)
        self.df_transform = self.drop_nan(self.df_transform)
        msno.matrix(self.df_transform, sparkline=False)
        
        self.one_hot_encoding()
        self.changing_data_type()
        self.df_transform = self.df_transform.drop(['Calculo_ERC'], axis=1) 
        self.df_transform = self.df_transform.drop(['Calculo_ERC_ALBUMINURIA'], axis=1)
        self.df_transform = self.df_transform.drop(['CALCULO TFG'], axis=1)
        self.df_transform = self.df_transform.drop(['CREATINURIA'], axis=1)
        self.df_transform = self.df_transform.drop(['ALBUMINURIA/CREATINURIA'], axis=1)
        self.scaling()
        #TODO: cast all df to float
        self.splitting()
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