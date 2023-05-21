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

        print(df['ENFERMEDADES'].unique())

        return df 
    
    def drugs_column_join(self, df):
        """
        Joins the following columns:
            - FARMACOS ANTIHIPERTENSIVOS
            - OTROS FARMACOS ANTIHIPERTENSIVOS
            - ANTIDIABETICOS
            - OTROS ANTIDIABETICOS
            - OTROS TRATAMIENTOS

        and removes all the 'NO APLICA', 'OTRO', 'NO', 'SIN DATO' values from the column farmacos, but only if they are not the only value in the column.

        Args:
            df: The pandas DataFrame.

        Returns:
            The pandas DataFrame with the joined and cleaned column farmacos.
        """

        # Join the columns.
        df['FARMACOS'] = df['FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['OTROS FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['ANTIDIABETICOS'] + ' + ' + df['OTROS ANTIDIABETICOS'] + ' + ' + df['OTROS TRATAMIENTOS']

        # Drop the original columns.
        df = df.drop(['FARMACOS ANTIHIPERTENSIVOS', 'OTROS FARMACOS ANTIHIPERTENSIVOS', 'ANTIDIABETICOS', 'OTROS ANTIDIABETICOS', 'OTROS TRATAMIENTOS'], axis=1)

        accepted_strings = {'NO APLICA':"",
                            'OTRO':"", 
                            'NO':"", 
                            'SIN DATO':""}

        df['FARMACOS'] = df['FARMACOS'].replace(accepted_strings)

        # Fill in any missing values with "NO APLICA".
        df['FARMACOS'] = df['FARMACOS'].fillna("NO APLICA")

        # Add IECA or ARA to the farmacos column if the patient receives that medication.
        for i, row in df.iterrows():
            if row['RECIBE IECA'] == 'SI' and 'IECA' not in row['FARMACOS']:
                df.at[i, 'FARMACOS'] += '+IECA'

                if row['RECIBE ARA II'] == 'SI' and 'ARA' not in row['FARMACOS']:
                    df.at[i, 'FARMACOS'] += '+ARA'

        # Drop the "RECIBE IECA" and "RECIBE ARA II" columns.
        df = df.drop(['RECIBE IECA', 'RECIBE ARA II'], axis=1)

        return df

    def run (self, df):
        self.load_data(df)
        print(self.eng_df.isna().any())
        self.eng_df = self.deseases_column_join(self.eng_df)
        self.eng_df = self.drugs_column_join(self.eng_df)

        return self.eng_df


