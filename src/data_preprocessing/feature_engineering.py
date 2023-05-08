import numpy as np

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
    
    def drugs_column_join(self, df):
        df['FARMACOS'] = df['FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['OTROS FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['ANTIDIABETICOS'] + ' + ' + df['OTROS ANTIDIABETICOS'] + ' + ' + df['OTROS TRATAMIENTOS']
        df = df.drop(['FARMACOS ANTIHIPERTENSIVOS', 'OTROS FARMACOS ANTIHIPERTENSIVOS', 'ANTIDIABETICOS', 'OTROS ANTIDIABETICOS', 'OTROS TRATAMIENTOS'], axis=1)
        df['FARMACOS'] = df['FARMACOS'].fillna("NO APLICA")

        #RECIBE IECA 
        for i, row in df.iterrows():
            if(((not 'IECA' in str(row['FARMACOS'])) and (str(row['RECIBE IECA']) == 'SI'))):
                df.at[i,'FARMACOS'] = str(df['FARMACOS'][i]) + '+IECA'

        df = df.drop(['RECIBE IECA'], axis=1)
        #RECIBE ARA II
        for i, row in df.iterrows():
            if(((not 'ARA' in str(row['FARMACOS'])) and (str(row['RECIBE ARA II']) == 'SI'))):
                df.at[i,'FARMACOS'] = str(df['FARMACOS'][i]) + '+ARA'

        df = df.drop(['RECIBE ARA II'], axis=1)
        
        return df 

    def run (self, df):
        self.load_data(df)
        print(self.eng_df.isna().any())
        self.eng_df = self.deseases_column_join(self.eng_df)
        self.eng_df = self.drugs_column_join(self.eng_df)

        return self.eng_df


