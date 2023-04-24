
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
            # Add the systems dictionary here
        }

        for system in systems:
            self.clean_df[f'{system}_1'] = col_coomorbilidad[0].apply(classify_system)
            self.clean_df[f'{system}_2'] = col_coomorbilidad[1].apply(classify_system) if 1 in col_coomorbilidad.columns else None
            self.clean_df[f'{system}_3'] = col_coomorbilidad[2].apply(classify_system) if 2 in col_coomorbilidad.columns else None

        return self.clean_df

    def run (self):
        self.eng_df = self.comorbilidad(self)

        return self.eng_df


