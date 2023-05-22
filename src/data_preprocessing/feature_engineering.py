import numpy as np
import re

class feature_eng:
    def __init__(self):
        self.eng_df = None

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
    
    def load_data(self, df):
        self.eng_df = df
        self.eng_df = self.data_types(self.eng_df)

   
    def deseases_column_join(self, df):
        df['ENFERMEDADES'] = df['OTROS DIAGNÓSTICOS'] + ' + ' + df['Coomorbilidad']
        df = df.drop(['Coomorbilidad', 'OTROS DIAGNÓSTICOS'], axis=1)

        return df 
    
    def join_and_clean_drugs_columns(self, df):
        df = df.fillna('')

        df['FARMACOS'] = df['FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['OTROS FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['ANTIDIABETICOS'] + ' + ' + df['OTROS ANTIDIABETICOS'] + ' + ' + df['OTROS TRATAMIENTOS']
        
        unwanted_phrases = ['NO APLICA', 'OTRO', 'NO', 'SIN DATO','PARA MANEJO NO FARMACOLÓGICO','PARA MANEJO NO FARMACOLOGICO','PARA MANEJO FARMACOLÓGICO', 'PARA MANEJO FARMACOLOGICO', 'OTROS FÁRMACOS', 'OTROS FARMACOS']
        
        # Split each 'FARMACOS' entry by ' + ' separator, filter out unwanted phrases, and join back with ' + ' separator
        df['FARMACOS'] = df['FARMACOS'].apply(lambda x: ' + '.join([word for word in x.split(' + ') if word.upper().strip() not in unwanted_phrases]))
        
        # Add 'IECA' or 'ARA' if the patient receives that medication
        for i, row in df.iterrows():
            if row['RECIBE IECA'] == 'SI' and 'IECA' not in row['FARMACOS']:
                df.at[i, 'FARMACOS'] += ' + IECA'
            if row['RECIBE ARA II'] == 'SI' and 'ARA' not in row['FARMACOS']:
                df.at[i, 'FARMACOS'] += ' + ARA'
        
        # Drop the original columns and 'RECIBE IECA', 'RECIBE ARA II'
        columns_to_drop = ['FARMACOS ANTIHIPERTENSIVOS', 'OTROS FARMACOS ANTIHIPERTENSIVOS', 'ANTIDIABETICOS', 'OTROS ANTIDIABETICOS', 'OTROS TRATAMIENTOS', 'RECIBE IECA', 'RECIBE ARA II']
        df.drop(columns_to_drop, axis=1, inplace=True)
        
        print(df['FARMACOS'].unique())
        return df

    def run (self, df):
        self.load_data(df)
        print(self.eng_df.isna().any())
        self.eng_df = self.deseases_column_join(self.eng_df)
        self.eng_df = self.join_and_clean_drugs_columns(self.eng_df)

        return self.eng_df


