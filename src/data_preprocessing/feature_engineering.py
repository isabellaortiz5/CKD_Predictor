
import pandas as pd

class feature_eng:
    def __init__(self, clean_df):
        self.clean_df = clean_df
        self.eng_df = None

    def comorbilidad(self):
        col_coomorbilidad = self.clean_df['Coomorbilidad']
        col_coomorbilidad = col_coomorbilidad.str.split(r'\s*\+\s*', expand=True)

        all_columns = pd.DataFrame()
        for col in col_coomorbilidad:
            all_columns = pd.concat([all_columns, col], axis=0)

        def classify_system(disease):
            for system, diseases in systems.items():
                if disease in diseases:
                    return system
            return 'unclassified'

        systems = {
            'cardiovascular': [
                'ACV', 'ARRITMIA CARDIACA', 'ACV (OBSTRUCCION DE VALVULA MITRAL)', 
                'ICC', 'CARDIOPATIA-ANTECEDENTE DE REVASCULARIZACION MIOCARDICA', 
                'IAM', 'ANTECEDENTE DE ACV', 'MARCAPASO'
            ],
            'endocrine': [
                'HIPOTIROIDISMO', 'DM INSULINODEPENDIENTE', 'HIPERGLICEMIA', 
                'DM', 'HIPERLIPIDEMIA', 'PREDIADETES'
            ],
            'renal': [
                'IRC', 'TRANSPLANTE RENAL', 'ERC'
            ],
            'respiratory': [
                'EPOC'
            ],
            'inmune': [
                'ANTECEDENTE DE CA DE MAMA', 'CA DE COLON', 'LINFOMA HODGKIN'
            ],
            'neurological': [
                'ALZHEIMER'
            ],
            'psychiatric': [
                'ANSIEDAD DEPRESION'
            ],
            'obesity': [
                'OBESIDAD'
            ],
            'thyroid': [
                'LESIONES NODULARES TIROIDEAS TIRADS 5', 'HIPOTIROIDISMO'
            ],
            'arthritis': [
                'ARTRITIS'
            ],
            'unclassified': [
                'SIN DATO', 'SIN CLASIFICAR', None
            ],
            'lipid metabolism': [
                'DISLIPIDEMIA', 'HIPERCOLESTEROLEMIA', 'LIPIDEMIA', 'DISLIPIDEMIAS'
            ]
        }

        for system in systems:
            self.clean_df[f'{system}_1'] = col_coomorbilidad[0].apply(classify_system)
            self.clean_df[f'{system}_2'] = col_coomorbilidad[1].apply(classify_system) if 1 in col_coomorbilidad.columns else None
            self.clean_df[f'{system}_3'] = col_coomorbilidad[2].apply(classify_system) if 2 in col_coomorbilidad.columns else None

        return self.clean_df
    def comorbilidad(self):
        col_coomorbilidad = self.clean_df['Coomorbilidad']
        col_coomorbilidad = col_coomorbilidad.str.split(r'\s*\+\s*', expand=True)

        all_columns = pd.DataFrame()
        for col in col_coomorbilidad:
            all_columns = pd.concat([all_columns, col], axis=0)      

        #df_temp = pd.DataFrame({'disease': all_columns})

        systems = {
            'cardiovascular': [
                'ACV', 'ARRITMIA CARDIACA', 'ACV (OBSTRUCCION DE VALVULA MITRAL)', 
                'ICC', 'CARDIOPATIA-ANTECEDENTE DE REVASCULARIZACION MIOCARDICA', 
                'IAM', 'ANTECEDENTE DE ACV'
            ],
            'endocrine': [
                'HIPOTIROIDISMO', 'DM INSULINODEPENDIENTE', 'HIPERGLICEMIA', 
                'DM', 'HIPERLIPIDEMIA'
            ],
            'renal': [
                'IRC', 'TRANSPLANTE RENAL', 'ERC'
            ],
            'respiratory': [
                'EPOC'
            ],
            'cancer': [
                'ANTECEDENTE DE CA DE MAMA', 'CA DE COLON', 'LINFOMA HODGKIN'
            ],
            'neurological': [
                'ALZHEIMER'
            ],
            'psychiatric': [
                'ANSIEDAD DEPRESION'
            ],
            'obesity': [
                'OBESIDAD'
            ],
            'thyroid': [
                'LESIONES NODULARES TIROIDEAS TIRADS 5', 'HIPOTIROIDISMO'
            ],
            'arthritis': [
                'ARTRITIS'
            ],
            'unclassified': [
                'SIN DATO', 'SIN CLASIFICAR', None
            ],
            'lipid metabolism': [
                'DISLIPIDEMIA', 'HIPERCOLESTEROLEMIA', 'LIPIDEMIA', 'DISLIPIDEMIAS'
            ],
            'other': [
                'PREDIADETES', 'MARCAPASO'
            ]
        }

        for system, diseases in systems.items():
            df_temp[system] = df_temp['disease'].apply(lambda x: 1 if x in diseases else 0)

        df_temp.drop(columns=['disease'], inplace=True)

        print(df_temp.info())

        return df_temp

    def run (self):
        self.eng_df = self.comorbilidad()

        return self.eng_df


