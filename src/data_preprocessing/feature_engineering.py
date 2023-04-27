

class feature_eng:
    def __init__(self, clean_df):
        self.clean_df = clean_df
        self.eng_df = None

    def deseases_column_join(self, df):
        df['ENFERMEDADES'] = df['OTROS DIAGNÓSTICOS'] + ' + ' + df['Coomorbilidad']
        df = df.drop(['Coomorbilidad', 'OTROS DIAGNÓSTICOS'], axis=1)
        return df 
    
    def drugs_column_join(self, df):
        df['FARMACOS'] = df['FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['OTROS FARMACOS ANTIHIPERTENSIVOS'] + ' + ' + df['ANTIDIABETICOS'] + ' + ' + df['OTROS ANTIDIABETICOS'] + ' + ' + df['OTROS TRATAMIENTOS']
        df = df.drop(['Coomorbilidad', 'OTROS DIAGNÓSTICOS'], axis=1)
        return df 

    def run (self):
        self.eng_df = self.deseases_column_join(self.clean_df)
        self.eng_df = self.drugs_column_join(self.clean_df)

        return self.eng_df


