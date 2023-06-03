import numpy as np
import pandas as pd
import missingno as msno
import re

paths = {
        "caqueta": '../../data/raw_2/caqueta_data_2.csv',
        "narino": '../../data/raw_2/Narino_data_2.csv',
        "putumayo": '../../data/raw_2/Putumayo_data_2.csv',
        "caqueta": '../../data/raw/caqueta_data.csv',
        "narino": '../../data/raw/Narino_data.csv',
        "putumayo": '../../data/raw/Putumayo_data.csv',
        "caqueta_2": '../../data/raw_2/caqueta_data_2.csv',
        "narino_2": '../../data/raw_2/Narino_data_2.csv',
        "putumayo_2": '../../data/raw_2/Putumayo_data_2.csv',
    }

common_drops = [
        "Año", "Mes", "Programa", "Evento", "Afiliados", "OrigenBD", "DesDepto", "CodMpio", "DescMpio",
        "Latitud_Y_Mpio", "Longitud_X_Mpio", "tipo_usuario", "Estado", "tipo_identifiCAcion", "Documento", "ConCAtenar", "nombre1",
        "nombre2", "apellido1", "apellido2", "FechaNac", "CiclosV", "DescrCiclosV", "QuinQ", "DescQuinQ","Género", "EnfoqueDif",
        "Hecho Victimizante", "RUV", "Nivel_Educativo", "Ocupación", "Tipo de afiliado", "Estado_Civil", "Discapacidad",
        "Grado de Discapacidad", "MUNICIPIO DONDE VIVE", "DIRECCIÓN DE DONDE VIVE", "TELEFONOS DE CONTACTO", "Zona",
        "Cód_poblado", "Nombre_poblado", "Latitud_Afiliado", "Longitud_Afiliado", "Validación_Dirección_Afiliado",
        "CodDepto_IPS", "DesDepto_IPS", "CodMpio_IPS", "DescMpio_IPS", "CodIPS", "Nombre_IPS", "Dirección_IPS",
        "Barrio_IPS", "Teléfono_IPS", "Latitud_IPS", "Longitud_IPS", "CONSUMO DE TABACO",
        "EL USUARIO CUENTA CON ATENCIÓN POR PARTE DEL EQUIPO MULTIDISCIPLINARIO DE LA SALUD ",
        "CONTROLADO/NO CONTROLADO",
        "PACIENTES CON 5 Ó MÁS MDCTOS FORMULADOS AL MES", "FECHA DE ENTREGA DE MEDICAMENTOS",
        "MODALIDAD DE ENTREGA DE MEDICAMENTOS",
        "PACIENTE I10X Y E119 QUE NO RECIBEN TRAMIENTO", "FECHA INGRESO AL PROGRAMA", "FECHA DEL SEGUNDO CONTROL",
        "FECHA DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL", "DX CONFIRMADO DE DIABETES MELLITUS",
        "FECHA DX CONFIRMADO DE DIABETES MELLITUS", "FECHA DIAGNÓSTICO DISLIPIDEMIAS", "TENSION SISTOLICA",
        "TENSION DIASTOLICA", "VALORACIÓN PODOLÓGICA (PIE EN DIABETICOS) POR MÉDICO GENERAL",
        "REALIZA ACTIVIDAD FISICA",
        "TABAQUISMO", "RESULTADO", "FACTORES DE RIESGO", "NUTRICIÓN", "TRABAJO SOCIAL", "MEDICINA INTERNA",
        "PSICOLOGIA",
        "NEFROLOGIA", "OFTALMOLOGÍA", "ENDOCRINOLOGIA", "CARDIOLOGIA", "ELECTROCARDIOGRAMA", "NEFROPROTECCIÓN",
        "TERAPIA SUSTITUCIÓN DIALÍTICA", "TALLER EDUCATIVO ENTREGA CARTILLAS", "FECHA_CLASIF_ERC",
        "FECHA DEL UROANALISIS",
        "FECHA DE ULTIMO SEGUIMIENTO ", "ETIOLOGIA DE LA ERC", "MODALIDAD COMO SE HACE EL SEGUIMIENTO DEL PACIENTE",
        "FECHA DE PRÓXIMO CONTROL ", "CAUSAS DE INASISTENCIA", "DISMINUYO/ AUMENTO ML"
        ]
narino_putumayo_drops = [
        "FECHA NUTRUCIÓN", "FECHA TRABAJO SOCIAL", "FECHA MEDICINA INTERNA", "FECHA PISCOLOGIA", "FECHA NEFROLOGIA",
        "FECHA OFTALMOMOGIA", "FECHA ENDOCRINOLOGIA", "FECHA CARDIOLOGIA", "FECHA ELECTROCARDIOGRAMA",
        "FECHA NEFROPROTECCIÓN"
        ]
narino_drops = [
        "REPETIDO"
        ]
putumayo_drops = [
        "FECHA DE TOMA DE CREATININA SÉRICA", "FECHA DE TOMA DE GLICEMIA 100 MG/DL_DIC",
        "FECHA DE TOMA DE COLESTEROL TOTAL > 200 MG/DL_DIC",
        "FECHA DE TOMA DE LDL > 130 MG/DL_DIC", "FECHA DE TOMA DE HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC",
        "FECHA DE TOMA DE ALBUMINURIA/CREATINURIA", "FECHA DE TOMA DE HEMOGLOBINA GLICOSILADA > DE 7%",
        "FECHA DE TOMA DE HEMOGRAMA",
        "FECHA DE TOMA DE POTASIO", "FECHA DE TOMA DE MICROALBINURIA", "FECHA DE TOMA DE CREATINURIA", "OBSERVACIONES",
        "FECHA DE TOMA DE TGD > 150 MG/DL_DIC","POTASIO", "HEMOGRAMA"
        ]


class Cleaning:

    def __init__(self, saving_path):
        global paths
        global common_drops
        global narino_putumayo_drops
        global narino_drops
        global putumayo_drops

        drops = [len(common_drops), len(narino_putumayo_drops), len(narino_drops), len(putumayo_drops)]
        print("Number of drops defined:")
        print("common_drops len: {}".format(len(common_drops)))
        print("narino_putumayo_drops len: {}".format(len(narino_putumayo_drops)))
        print("narino_drops len: {}".format(len(narino_drops)))
        print("putumayo_drops len: {}".format(len(putumayo_drops)))
        print("total drops: {}".format(sum(drops)))

        self.saving_path = saving_path
        self.caqueta_df = None
        self.narino_df = None
        self.putumayo_df = None
        self.first_caqueta_clean = None
        self.first_putumayo_clean = None
        self.first_narino_clean = None
        self.df = None
        self.unified_df = None
        self.df_after_fix = None
        self.df_clean = None

    def read_csv(self):
        self.caqueta_df = pd.read_csv(paths["caqueta"],low_memory=False)
        self.narino_df = pd.read_csv(paths["narino"],low_memory=False)
        self.putumayo_df = pd.read_csv(paths["putumayo"],low_memory=False)
        self.caqueta_df_2 = pd.read_csv(paths["caqueta_2"],low_memory=False)
        self.narino_df_2 = pd.read_csv(paths["narino_2"],low_memory=False)
        self.putumayo_df_2 = pd.read_csv(paths["putumayo_2"],low_memory=False)

    def drop_columns(self):
        caqueta_df_clean = self.caqueta_df.drop(common_drops, axis=1)
        caqueta_df_clean = self.caqueta_df_2.drop(common_drops, axis=1)

        narino_df_clean = self.narino_df.drop(common_drops, axis=1)
        narino_df_clean = self.narino_df_2.drop(common_drops, axis=1)
        narino_df_clean['Edad'] = self.narino_df['Edad']
        narino_df_clean = narino_df_clean.drop(narino_drops, axis=1)
        narino_df_clean = narino_df_clean.drop(narino_putumayo_drops, axis=1)

        putumayo_df_clean = self.putumayo_df.drop(common_drops, axis=1)
        putumayo_df_clean = self.putumayo_df_2.drop(common_drops, axis=1)
        putumayo_df_clean = putumayo_df_clean.drop(putumayo_drops, axis=1)
        putumayo_df_clean = putumayo_df_clean.drop(narino_putumayo_drops, axis=1)

        self.putumayo_df = putumayo_df_clean
        self.narino_df = narino_df_clean
        self.caqueta_df = caqueta_df_clean

        self.first_caqueta_clean = caqueta_df_clean
        self.first_putumayo_clean = putumayo_df_clean
        self.first_narino_clean = narino_df_clean

    def concat_dfs(self):
        self.df = pd.concat([self.putumayo_df, self.narino_df, self.caqueta_df], axis=0)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.unified_df = self.df

    def replace_blanks(self):
        self.df = self.df.rename(columns=lambda x: x.strip())
        self.df = self.df.replace(r'^\s*$', np.nan, regex=True)
        self.df["FechaNovedadFallecido"] = self.df["FechaNovedadFallecido"].fillna('no aplica')
        self.df["Coomorbilidad"] = self.df["Coomorbilidad"].fillna('no aplica')
        self.df["Condición de Discapacidad"] = self.df["Condición de Discapacidad"].fillna('no aplica')
        self.df["Tipo de Discapacidad"] = self.df["Tipo de Discapacidad"].fillna('no aplica')
        self.df["OTROS ANTIDIABETICOS"] = self.df["OTROS ANTIDIABETICOS"].fillna('no aplica')
        self.df["OTROS TRATAMIENTOS"] = self.df["OTROS TRATAMIENTOS"].fillna('no aplica')
        self.df["OTROS FARMACOS ANTIHIPERTENSIVOS"] = self.df["OTROS FARMACOS ANTIHIPERTENSIVOS"].fillna('no aplica')
        self.df = self.df.replace("#DIV/0!", np.nan)
        self.df = self.df.replace("#NUM!", np.nan)
        self.df = self.df.replace("#VALUE!", np.nan)

    def data_types(self):
        self.df['Grupo de Riesgo'].astype('object').dtypes
        self.df['CodDepto'].astype('float64').dtypes
        self.df['FechaNovedadFallecido'].astype('object').dtypes
        self.df['Edad'].astype('float64').dtypes
        self.df['Cod_Género'].astype('float64').dtypes
        self.df['Tipo de Discapacidad'].astype('object').dtypes
        self.df['Condición de Discapacidad'].astype('object').dtypes
        self.df['Pertenencia Étnica'].astype('object').dtypes
        self.df['Coomorbilidad'].astype('object').dtypes
        self.df['ADHERENCIA AL TRATAMIENTO'].astype('object').dtypes
        self.df['Fumador Activo'].astype('float64').dtypes
        self.df['CONSUMO DE ALCOHOL'].astype('object').dtypes
        self.df['ENTREGA DE MEDICAMENTO OPORTUNA'].astype('object').dtypes
        self.df['FARMACOS ANTIHIPERTENSIVOS'].astype('object').dtypes
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].astype('object').dtypes
        self.df['RECIBE IECA'].astype('object').dtypes
        self.df['RECIBE ARA II'].astype('object').dtypes
        self.df['ESTATINA'].astype('object').dtypes
        self.df['ANTIDIABETICOS'].astype('object').dtypes
        self.df['OTROS ANTIDIABETICOS'].astype('object').dtypes
        self.df['OTROS TRATAMIENTOS'].astype('object').dtypes
        self.df['OTROS DIAGNÓSTICOS'].astype('object').dtypes
        self.df['PESO'].astype('float64').dtypes
        self.df['TALLA'].astype('float64').dtypes
        self.df['IMC'].astype('float64').dtypes
        self.df['OBESIDAD'].astype('object').dtypes
        self.df['CALCULO DE RIESGO DE Framingham (% a 10 años)'].astype('float64').dtypes
        self.df['Clasificación de RCV Global'].astype('object').dtypes
        self.df['DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL'].astype('float64').dtypes
        self.df['CÓD_DIABETES'].astype('float64').dtypes
        self.df['CLASIFICACION DIABETES'].astype('object').dtypes
        self.df['DIAGNÓSTICO DISLIPIDEMIAS'].astype('object').dtypes
        self.df['CÓD_ANTEDECENTE'].astype('float64').dtypes
        self.df['PRESION ARTERIAL'].astype('object').dtypes
        self.df['COLESTEROL ALTO'].astype('object').dtypes
        self.df['HDL ALTO'].astype('object').dtypes
        self.df['CLASIFICACIÓN DE RIESGO CARDIOVASCULAR'].astype('object').dtypes
        self.df['CALCULO TFG'].astype('float64').dtypes
        self.df['CLASIFICACIÓN ESTADIO'].astype('object').dtypes
        self.df['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC'].astype('float64').dtypes
        self.df['GLICEMIA 100 MG/DL_DIC'].astype('float64').dtypes
        self.df['COLESTEROL TOTAL > 200 MG/DL_DIC'].astype('float64').dtypes
        self.df['LDL > 130 MG/DL_DIC'].astype('float64').dtypes
        self.df['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'].astype('float64').dtypes
        self.df['TGD > 150 MG/DL_DIC'].astype('float64').dtypes
        self.df['ALBUMINURIA/CREATINURIA'].astype('float64').dtypes
        self.df['HEMOGLOBINA GLICOSILADA > DE 7%'].astype('float64').dtypes
        self.df['MICROALBINURIA'].astype('float64').dtypes
        self.df['CREATINURIA'].astype('float64').dtypes
        self.df['UROANALIS'].astype('object').dtypes
        self.df['PERIMETRO ABDOMINAL'].astype('float64').dtypes
        self.df['Complicación Cardiaca'].astype('object').dtypes
        self.df['Complicación Cerebral'].astype('object').dtypes
        self.df['Complicación Retinianas'].astype('object').dtypes
        self.df['Complicación Vascular'].astype('object').dtypes
        self.df['Complicación Renales'].astype('object').dtypes

    @staticmethod
    def trim_all_columns(df):
        trim_strings = lambda x: x.strip() if isinstance(x, str) else x
        return df.applymap(trim_strings)
    
    def fix_plus_sign(self,df):
        df = df.replace('-', '+', regex=True)
        df = df.replace(r'\s*\+\s*', '+', regex=True)
        
        return df

    def fixed_data(self):

        self.df = self.fix_plus_sign(self.df)
        self.df = self.trim_all_columns(self.df)

        #df upper case
        self.df = self.df.apply(lambda x: x.astype(str).str.upper())
        
        #ldl > 130
        self.df['LDL > 130 MG/DL_DIC'] = self.df['LDL > 130 MG/DL_DIC'].str.replace('1845\+01\+01', '0', regex=True)
        self.df['LDL > 130 MG/DL_DIC'] = self.df['LDL > 130 MG/DL_DIC'].str.replace('4*9', '9', regex=True)
        self.df['LDL > 130 MG/DL_DIC'] = self.df['LDL > 130 MG/DL_DIC'].str.replace('14*28', '28', regex=True)
        self.df['LDL > 130 MG/DL_DIC'] = self.df['LDL > 130 MG/DL_DIC'].str.replace('14\*9', '14', regex=True)
        self.df['LDL > 130 MG/DL_DIC'] = self.df['LDL > 130 MG/DL_DIC'].str.replace('109\+', '109', regex=True)

        # comas por puntos en float
        comma_col_list = [ 'CodDepto', 'Edad', 'Cod_Género', 'Fumador Activo', 'PESO', 'TALLA', 'IMC', 
                        'CALCULO DE RIESGO DE Framingham (% a 10 años)', 'DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL', 
                        'CÓD_DIABETES', 'CÓD_ANTEDECENTE', 'CALCULO TFG', 'CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC', 'GLICEMIA 100 MG/DL_DIC', 
                        'COLESTEROL TOTAL > 200 MG/DL_DIC', 'LDL > 130 MG/DL_DIC', 'HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC', 'TGD > 150 MG/DL_DIC', 'ALBUMINURIA/CREATINURIA', 
                        'HEMOGLOBINA GLICOSILADA > DE 7%','MICROALBINURIA', 'CREATINURIA', 'PERIMETRO ABDOMINAL' ]

        for col_name in comma_col_list:
            self.df[col_name] = self.df[col_name].str.replace(',', '.', regex=True)
        
        #Pertenencia Étnica
        self.df['Pertenencia Étnica'] = self.df['Pertenencia Étnica'].str.upper()

        #comorbilidad
        self.df['Coomorbilidad'] = self.df['Coomorbilidad'].str.replace('IHPOTIROIDISMO', 'HIPOTIROIDISMO', regex=True)
        self.df['Coomorbilidad'] = self.df['Coomorbilidad'].str.replace('HIPOTIROIDIDMO', 'HIPOTIROIDISMO', regex=True)
        self.df['Coomorbilidad'] = self.df['Coomorbilidad'].str.replace('IRC', 'ERC', regex=True)
        self.df['Coomorbilidad'] = self.df['Coomorbilidad'].str.replace('ERCE5', 'ERC', regex=True)
        self.df['Coomorbilidad'] = self.df['Coomorbilidad'].str.replace('ANSIEDAD DEPRESION', 'ANSIEDAD+DEPRESION', regex=True)
        self.df['Coomorbilidad'] = self.df['Coomorbilidad'].str.replace('PREDIADETES', 'PREDIABETES', regex=True)
        
        #Creatinina Sérica
        self.df['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC'] = self.df['CREATININA SÉRICA (HOMBRES > 1.7 MG/DL - MUJERES > 1.4 MG/DL) _DIC'].str.replace('1..02', '1.02', regex=True)
        
        #'HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'
        self.df['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'] = self.df['HDL HOMBRE - 40 MG/DL Y HDL MUJER - 50 MG/DL_DIC'].str.replace('46\+', '46', regex=True)

        #'TGD > 150 MG/DL_DIC'
        self.df['TGD > 150 MG/DL_DIC'] = self.df['TGD > 150 MG/DL_DIC'].str.replace('260\+80', '260', regex=True)

        #'HEMOGLOBINA GLICOSILADA > DE 7%'
        self.df['HEMOGLOBINA GLICOSILADA > DE 7%'] = self.df['HEMOGLOBINA GLICOSILADA > DE 7%'].str.replace('28\+07\+2022', '0', regex=True)

        #microalbinuria
        self.df['MICROALBINURIA'] = self.df['MICROALBINURIA'].str.replace('>300', '300', regex=True)

        #creatinuria
        self.df['CREATINURIA'] = self.df['CREATINURIA'].str.replace('1845\+01\+01', '0', regex=True)
        
        #OTROS DIAGNÓSTICOS
        self.df['OTROS DIAGNÓSTICOS'] = self.df['OTROS DIAGNÓSTICOS'].str.replace(', NO ESPECIFICADO', '', regex=True)
        self.df['OTROS DIAGNÓSTICOS'] = self.df['OTROS DIAGNÓSTICOS'].str.replace(', NO ESPECIFICADA', '', regex=True)
        self.df['OTROS DIAGNÓSTICOS'] = self.df['OTROS DIAGNÓSTICOS'].str.replace(', ASI DESCRITA', '', regex=True)
        self.df['OTROS DIAGNÓSTICOS'] = self.df['OTROS DIAGNÓSTICOS'].str.replace(' DEBIDA A EXCESO DE CALORIAS', '', regex=True)
        self.df['OTROS DIAGNÓSTICOS'] = self.df['OTROS DIAGNÓSTICOS'].str.replace(', SIN OTRA ESPECIFICACION', '', regex=True)

        #FARMACOS ANTIHIPERTENSIVOS
        self.df['FARMACOS ANTIHIPERTENSIVOS'] = self.df['FARMACOS ANTIHIPERTENSIVOS'].str.replace(' O ', '+', regex=True)
        self.df['FARMACOS ANTIHIPERTENSIVOS'] = self.df['FARMACOS ANTIHIPERTENSIVOS'].str.replace(' Ó ', '+', regex=True)
        self.df['FARMACOS ANTIHIPERTENSIVOS'] = self.df['FARMACOS ANTIHIPERTENSIVOS'].str.replace(' UNICAMENTE', '', regex=True)
        self.df['FARMACOS ANTIHIPERTENSIVOS'] = self.df['FARMACOS ANTIHIPERTENSIVOS'].str.replace('HIDROCLOROTIAZIDA (HCTZ)', 'HCTZ', regex=True)
        self.df['FARMACOS ANTIHIPERTENSIVOS'] = self.df['FARMACOS ANTIHIPERTENSIVOS'].str.replace('HIDROCLOROTIAZIDA', 'HCTZ', regex=True)
        self.df['FARMACOS ANTIHIPERTENSIVOS'] = self.df['FARMACOS ANTIHIPERTENSIVOS'].str.replace('HCTZ (HCTZ)', 'HTCZ', regex=True)

        #OTROS FARMACOS ANTIHIPERTENSIVOS
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('/', '+', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace(',', '+', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace(r'\bNO\b', 'NO APLICA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace(r'\s*(?:1|2|MG|MGR|0|3|5|7|8|10|20|25|40|50|80|100|200)\s*', ' ', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace(r'\sX\s', '', regex=True)        
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace(r'\sX$', '', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace(r'\+$', '', regex=True)
        self.df = self.trim_all_columns(self.df)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('NO APLICA APLICA', 'NO APLICA' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('METROPROLOL', 'METOPROLOL' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ROSUBASTATINA', 'ROSUVASTATINA' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('SACUBITRILO+VVALSARTAN', 'SACUBITRILO+VALSARTAN' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('GNFIBROXILO', 'GEMFIBROZILO' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('GENFIBROZILO+ATROVASTATINA+EZOMEPRAZOL', 'GEMFIBROZILO+ATROVASTATINA+EZOMEPRAZOL' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('METFORMINA+ROSUVASTATIBNA', 'METFORMINA+ROSUVASTATINA' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VIDAGLIPTINA', 'VILDAGLIPTINA' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('METOPROLOL   COMPRIMIDO', 'METOPROLOL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('CLOGIDRATO', 'CLORIDRATO' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ATORVASTAINA', 'ATORVASTATINA' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN METOPROLOL', 'LOSARTAN+METOPROLOL' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VILDAGLIPTINA', 'VIDALGLIPTINA' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN . ACETIL SALICILICO', 'LOSARTAN+ACETIL SALICILICO' , regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN   HCT', 'LOSARTAN+HCT')
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN HCT', 'LOSARTAN+HCT')
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTANX', 'LOSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSATRAN', 'LOSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARATAN', 'LOSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSATAN', 'LOSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('KOSARTAN', 'LOSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VALSARTAN HCT', 'VALSARTAN+HCT', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('NEBIDOLOL', 'NEBIVOLOL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ACIDO ACETIL SALICILICO   M', 'ACIDO ACETILSALICILICO', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALAPRIL METFORMINA', 'ENALAPRIL+METFORMINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ATORVASTATINA   Y LEVO', 'ATORVASTATINA+LEVO', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALPRIL', 'ENALAPRIL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSAR HCT AMLODIPINO ASA', 'LOSARTAN+HCT+AMLODIPINO+ASA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ATORVATATINA', 'ATORVASTATINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN+HCT ASA', 'LOSARTAN+HCT+ASA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VALSARTAN6', 'VALSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VIDALGLIPTINAX', 'VIDALGLIPTINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALAPRIL   LEVOTIROX', 'ENALAPRIL+LEVOTIROXINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('INS GLARGINA', 'INSULINA GLARGINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('INSULINA GLARGINA+NIDAGLIPTINA+ESPIRINOLATONA+LEVOT', 'INSULINA GLARGINA+VILDAGLIPTINA+ESPIRONOLACTONA+LEVOT', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALAPRIL HCT', 'ENALAPRIL+HCT', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALAPRIL  INS DEGLUDEC+DEPAGLIFOZINA', 'ENALAPRIL+INSULINA DEGLUDEC+DEPAGLIFLOZINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN AMLOD ASA HCT', 'LOSARTAN+AMOLDIPINO+ASA+HCT', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALAPRIL MALEATOX', 'ENALAPRIL+MALEATO', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('INSULINA GLARGINA VIDALGLIPTINA', 'INSULINA+GLARGINA+VIDALGLIPTINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN   METFORMINA', 'LOSARTAN+METFORMINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('METFORMINAX    VLDAGLIPTINA', 'METFORMINA+VIDALGLIPTINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN  Y METFORMINA', 'LOSARTAN+METFORMINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('INS DEGLUDEC', 'INSULINA DEGLUDEC', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN+HCT+MEFT', 'LOSARTAN+HCT+METFORMINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('ENALAPRIL  INSULINA DEGLUDEC+DEPAGLIFOZINA', 'ENALAPRIL+INSULINA DEGLUDEC+DEPAGLIFOZINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VALSARTANX 6', 'VALSARTAN', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('METROPOLOL', 'METOPROLOL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('IBERSERTAN+AMLOD', 'IBERSERTAN+AMLODIPINO', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('HIDROCLOROTIAZIDA   TABLETA', 'HIDROCLOROTIAZIDA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('METOPROLOL UNICAMENTE', 'METOPROLOL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('AMLODIPINO+GHIDROCLORORIZIDA Y METFORMINA', 'AMLODIPINO+HIDROCLOROTIAZIDA+METFORMINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('VILDAGLITINA METFORMINA', 'VILDAGLIPTINA+METFORMINA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('PROPONOOL', 'PROPRANOLOL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN+CARVEDILLOL+ASA+HTZA+CLOPIDROGEL', 'LOSARTAN+CARVEDILOL+ASA+HTZA+CLOPIDOGREL', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('LOSARTAN+ESPIRINOLACTONA', 'LOSARTAN+ESPIRONOLACTONA', regex=True)
        self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('HOLMENTAN H', 'HOLMENTAN', regex=True)


        #self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].str.replace('\s+', ' ')
        #fix plus signs
        self.df = self.fix_plus_sign(self.df)

    def fixed_data2(self):
        print("Unique values in column:")
        print("**{}**".format(self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].unique()))
        
        self.df = self.trim_all_columns(self.df)

        patterns_and_replacements = [
            (r'\bLOSARTAN\s*\+\s*HCT\s*ASA\b', 'LOSARTAN+HCT+ASA'),
            (r'\bLOSARTAN\s*\+\s*HCTA\s*ASA\b', 'LOSARTAN+HCT+ASA'),
            (r'\bLOSARTAN\s*AMLODIPINO\b', 'LOSARTAN+AMLODIPINO'),
            (r'\bENALAPRIL\s*AMLODIPINO\s*HCT\b', 'ENALAPRIL+AMLODIPINO+HCT'),
            (r'\bVALSARTAN\s*Y\s*AMLODIPINO\b', 'VALSARTAN+AMLODIPINO'),
            (r'\bVALSARTAN\s+6\b', 'VALSARTAN'),
            (r'\bHIDROCLOROTIAZIDA\b', 'HCTZ'),
            (r'\bHCTZ\s*+\s*ENALAPRIL\s*+\s*NIFEDIPINO\b', 'HCTZ+ENALAPRIL+NIFEDIPINO'),
            (r'\bLOSARTAN\s*\.\s*FUROSEMIODA\s*\+\s*AMLODIPINO\b', 'LOSARTAN+FUROSEMIDA+AMLODIPINO'),
            (r'\bINSULINA DEGLUDEC\s*\+\s*DEPAGLIFOZINA\b', 'INSULINA DEGLUDEC+DAPAGLIFLOZINA'),
            (r'\s{2,}', ' '),  # Replace multiple spaces with single space
            (r'\s*-\s*', '-'),  # Remove spaces around hyphens
            (r'\s*\/\s*', '/'),  # Remove spaces around slashes
            (r'\s*\+\s*', '+'),  # Remove spaces around plus signs
            (r'(\b[a-zA-Z])\.([a-zA-Z])\.', r'\1\2'),  # Remove periods from abbreviations
            (r'\s*\.+\s*', '.'),  # Remove multiple periods and spaces around periods
            (r'\s*\,\s*', ','),  # Remove spaces around commas
             (r'(\b\w+)\.(\w+\b)', r'\1\2'),  # Remove periods between words
            (r'\bTELMISARTAN\b', 'TELMISARTAN'),
            (r'\bHOLMENTAN\b', 'OLMESARTAN'),
            (r'\bLOSRTAN\b', 'LOSARTAN'),
            (r'\bLOSAR\b', 'LOSARTAN'),
            (r'\bLOSARTSN\b', 'LOSARTAN'),
            (r'\bLIBERSARTAN\b', 'IRBESARTAN'),
            (r'\bIBERSERTAN\b', 'IRBESARTAN'),
            (r'\bIBERSARTAN\b', 'IRBESARTAN'),
            (r'\bANALAPRIL\b', 'ENALAPRIL'),
            (r'\bCARVEDILLOL\b', 'CARVEDILOL'),
            (r'\bENLAPRIL\b', 'ENALAPRIL'),
            (r'\bENELAPRIL\b', 'ENALAPRIL'),
            (r'\bENALAPRIL\s+INSULINA\b', 'ENALAPRIL+INSULINA'),
            (r'\bSACUBITRILO\+VVALSARTAN\b', 'SACUBITRILO+VALSARTAN'),
            (r'\bESPIRINOLACTINA\b', 'ESPIRONOLACTONA'),
            (r'\bESPIRINOLATONA\b', 'ESPIRONOLACTONA'),
            (r'\bMETFORMINA\s*+\s*ROSUVASTATIBNA\b', 'METFORMINA+ROSUVASTATINA'),
            (r'\bATORVASRTATINA\b', 'ATORVASTATINA'),
            (r'\bGENGIBROZILO\b', 'GEMFIBROZILO'),
            (r'\bGENFIBROZILO\b', 'GEMFIBROZILO'),
            (r'\bGENFIBROXILO\b', 'GEMFIBROZILO'),
            (r'\bLIRAGLUTINA\b', 'LIRAGLUTIDE'),
            (r'\bCARVERIDOL\b', 'CARVEDILOL'),
            (r'\bCARDERIDOL\b', 'CARVEDILOL'),
            (r'\bNIMODIFPINO\b', 'NIMODIPINO'),
            (r'\bIECA\b', 'IECA'),
            (r'\bARA\b', 'ARA'),
            (r'\bHCTZ\b', 'HCTZ'),
            (r'\bHCT\b', 'HCTZ'),
            (r'\bMEFT\b', 'METFORMINA'),
            (r'\bMALEATO\b', 'MALEATO'),
            (r'\bDEGLUDEC\b', 'DEGLUDEC'),
            (r'\bEUTIROX\b', 'LEVO'),
            (r'\bLEVOT\b', 'LEVO'),
            (r'\bLEVO\b', 'LEVOTIROXINA'),
            (r'\bINSULINA DEGLUDEC+DAPAGLIFLOZIN\b', 'INSULINA DEGLUDEC+DAPAGLIFLOZIN'),
            (r'\bAMLODIPINO\s+GHIDROCLORORIZIDA\b', 'AMLODIPINO+HIDROCLOROTIAZIDA'),
            (r'\bGEMFIBROZILO\b', 'GEMFIBROZIL'),
            (r'\bGENFIBROZILO\b', 'GEMFIBROZIL'),
            (r'\bGLIMEPERIDINA\b', 'GLIMEPERIDINA'),
            (r'\bGLIMEPRIDINA\b', 'GLIMEPERIDINA'),
            (r'\bNO APLICA\b', 'NO APLICA'),
            (r'\bTRATAMIENTO NO APLICA FARMACOLOGICO\b', 'NO APLICA'),
            (r'\bRETIRADO\b', 'NO APLICA'),
            (r'\bFUROSEMIDA\b', 'FUROSEMIDA'),
            (r'\bFLOXETINA\b', 'FLOXETINA'),
            (r'\bWARFARINA\b', 'WARFARINA'),
            (r'\bBLOQUEADOR BETA\b', 'BETA BLOQUEADOR'),
            (r'\bPROPRANOLOL\b', 'PROPRANOLOL'),
            (r'\bNEBIVOLOL\b', 'NEBIVOLOL'),
            (r'\bNIDAGLIPTINA\b', 'NIDAGLIPTINA'),
            (r'\bINSULINA GLARGINA\b', 'INSULINA GLARGINA'),
            (r'\bSITAGLIPTINA\b', 'SITAGLIPTINA'),
            (r'\bVIDALGLIPTINA\b', 'VIDALGLIPTINA'),
            (r'\bVILDAGLIPTINA\b', 'VILDAGLIPTINA'),
            (r'\bDAPAGLIFLOZINA\b', 'DAPAGLIFLOZINA'),
            (r'\bEMPAGLIFLOZINA\b', 'EMPAGLIFLOZINA'),
            (r'\bLANSOPRAZOL\b', 'LANSOPRAZOL'),
            (r'\bEZOMEPRAZOL\b', 'ESOMEPRAZOL'),
            (r'\bCLOPIDOGREL\b', 'CLOPIDOGREL'),
            (r'\bGLIMEPERIDINA\b', 'GLIMEPIRIDA'),
            (r'\bGLIMEPIRIDE\b', 'GLIMEPIRIDA'),
            (r'\bACIDO ACETILSALICILICO\b', 'ASA'),
            (r'\bAMLOD\b', 'AMLODIPINO'),            
            (r'\bMETOCARBAMOL\b', 'METHOCARBAMOL'),
            (r'\bMETFORMINA\s*+\s*ROSUVASTATIBNA\b', 'METFORMINA+ROSUVASTATINA'),
            (r'\bOTORVASTATINA\b', 'ATORVASTATINA'),
            (r'\bVALSARTAN\s*X\b', 'VALSARTAN'),
            (r'\bVALSARTAN\s*\*\s*6\b', 'VALSARTAN'),
            (r'\bARA\s*II\b', 'ARA'),
            (r'\bARA\s*\+\s*AMLODIPINO\b', 'ARA+AMLODIPINO'),
            (r'\bARA\s*\+\s*NIFEDIPINO\b', 'ARA+NIFEDIPINO'),
            (r'\bLOSARTAN HCTZ\b', 'LOSARTAN+HCTZ'),
            (r'\bARA\s*\+\s*HIDROCLOROTIAZIDA\b', 'ARA+HCTZ'),
            (r'\bARA\s*\+\s*METOPROLOL\b', 'ARA+METOPROLOL'),
            (r'\bVALSARTAN\+HCTZTABLETA\b', 'VALSARTAN+HCTZ'),
            (r'\bAMLODIPINO TABLETA\b', 'AMLODIPINO'),
            (r'\bVALSARTAN\+HCTZTABLETA\b', 'VALSARTAN+HCTZ'),
            (r'\bLOSARTAN ASA ROSUV\b', 'LOSARTAN+ASA+ROSUVASTATINA'),
            (r'\bAMLODIPINO\+LOSARAN\.\+HCTZ\+CLONIDINA\b', 'AMLODIPINO+LOSARANTAN+HCTZ+CLONIDINA'),
            (r'\bAMLODIPINO\+GHIDROCLORORIZIDA Y METFORMINA\b', 'AMLODIPINO+HCTZ+METFORMINA'),
            (r'\bIRBESARTAN\+HCTZ\.\b', 'IRBESARTAN+HCTZ'),
            (r'\bMIOCARDIX\+TELMISARTAN\b', 'MICARDIS+TELMISARTAN'),
            (r'\bLOSARTAN AMLODIPINI ASA\b', 'LOSARTAN+AMLODIPINO+ASA'),
            (r'\bENALAPRIL ASA\b', 'ENALAPRIL+ASA'),
            (r'\bACETIL SALICILICO\b', 'ASA'),
            (r'\bMELOXICAN\b', 'MELOXICAM'),
            (r'\bROSUVASTATIBNA\b', 'ROSUVASTATINA'),
            (r'\bDAPAGLIFOXINA\b', 'DAPAGLIFLOXINA'),
            (r'\bRIVAROXON\b', ' RIVAROXABAN'),
            (r'\bFLOXETINA\b', 'FLUOXETINA'),
            (r'\bACETILSALICILICO\b', 'ASA'),
            (r'\bMTOPROLOL\b', 'METOPROLOL'),
            (r'\bESPIRONOLCTONA\b', 'ESPIRONOLACTONA'),
            (r'\bVILDAGLIPTIMA\b', 'VIDALGLIPTINA'),
            (r'\bMETFORMINA\+VILDAGLIPTIMA\+METFOTMINA\b', 'METFORMINA+VIDALGLIPTINA'),
            (r'\bGLIMEPIRIDA\+ATORVASTATINA\+VIDALGLIPTINA R\+METFORMINA R\b', 'GLIMEPIRIDA+ATORVASTATINA+VIDALGLIPTINA+METFORMINA'),
            (r'\bCARBAMEZAPINA\b', 'CARBAMAZEPINA'),
            (r'\bEMPAGLIFOZINA\b', 'EMPAGLIFLOZIN'),
            (r'\bHCTZ\.\b', 'HCTZ'),
            (r'\s*TABLETA\b', '')

        ]

        for pattern, replacement in patterns_and_replacements:
            compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)
            self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'] = self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].apply(
                lambda x: re.sub(compiled_pattern, replacement, x) if isinstance(x, str) else x
            )

        print("Fixed unique values in column:")
        print("**{}**".format(self.df['OTROS FARMACOS ANTIHIPERTENSIVOS'].unique()))

        return self.df


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

    def drop_nan(self):
        nan_percentages = self.get_nan_per_col(self.df)
        self.df_clean, columns_dropped = self.remove_by_nan(5, nan_percentages, self.df)
        self.df_clean = self.df_clean.dropna()

    def save_df(self):
        self.df_clean = self.df
        self.df_clean = self.df_clean.reset_index(drop=True)
        self.df_clean.to_csv(self.saving_path)
        print("Clean data successfully saved in: {}".format(self.saving_path))

    def run(self):
        print("------------------------------------------------")
        print("Cleaning...")
        self.read_csv()
        self.drop_columns()
        self.concat_dfs()
        self.replace_blanks()
        self.fixed_data()
        self.fixed_data2()
        self.data_types()
        self.drop_nan()
        print("Data successfully cleaned!")
        self.save_df()
        print("------------------------------------------------")

    #gets

    def get_caqueta_df(self):
        return self.caqueta_df

    def get_narino_df(self):
        return self.narino_df

    def get_putumayo_df(self):
        return self.putumayo_df

    def get_first_caqueta_clean(self):
        return self.first_caqueta_clean

    def get_first_narino_clean(self):
        return self.first_narino_clean

    def get_first_putumayo_clean(self):
        return self.first_putumayo_clean

    def get_df(self):
        return self.df
    
    def get_unified_df(self):
        return self.unified_df
    
    
    def get_df_clean(self):
        return self.df_clean

"""

# Plot missingness of unified data
print("********* UNIFIED DF ********")
print(df.info())
msno.matrix(df);
plt.title("Data after main drops and blanks to no aplica")
plt.show()

# Plot correlation heatmap of missingness
msno.heatmap(df, cmap='rainbow');
plt.title("Missingness - Data after main drops")
#plt.rc('font', size=1)
plt.show()

print("************************* Imputing with MICE ********************")

df_mice = df.filter(['CALCULO DE RIESGO DE Framingham (% a 10 años)','GLICEMIA 100 MG/DL_DIC','COLESTEROL TOTAL > 200 MG/DL_DIC', 'PERIMETRO ABDOMINAL'], axis=1).copy()

# Define MICE Imputer and fill missing values
mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending')

df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)

print("************************* imputed ********************")
print(df_mice_imputed.info())
df['CALCULO DE RIESGO DE Framingham (% a 10 años)'] = df_mice_imputed['CALCULO DE RIESGO DE Framingham (% a 10 años)']
df['GLICEMIA 100 MG/DL_DIC'] = df_mice_imputed['GLICEMIA 100 MG/DL_DIC']
df['COLESTEROL TOTAL > 200 MG/DL_DIC'] = df_mice_imputed['COLESTEROL TOTAL > 200 MG/DL_DIC']
df['PERIMETRO ABDOMINAL'] = df_mice_imputed['PERIMETRO ABDOMINAL']

msno.matrix(df);
plt.title("Missingness Data after imputation")
plt.show()

def get_nan_per_col(df):
    
      Iterates trough every column and caculates the percentage of nan values present in each column.

      :param df: {pandas data frame}.
      :type df: DataFrame.
      :return: a series of all the nan percentages asociated to the column name.
      :rtype: series.

  
    NAN_percentage = ((df.isna().sum() * 100) / df.shape[0]).sort_values(ascending=True)
    return NAN_percentage


def removeByNan(accepted_nan_percentage, nan_percentage_series, df):
    
      Takes a series that contains all the columns name with the respective nan percentage,
      then drops the columns that had nan content above the diven max nan limit.

      :param accepted_nan_percentage: {positive number that represent the limit of nan % acceptance}.
      :param nan_percentage_series: {series that contains all the columns name with the respective nan percentage}.
      :param df: {pandas data frame}.
      :type accepted_nan_percentage: float.
      :type nan_percentage_series: series.
      :type df: DataFrame.
      :return df: A dataframe without the columns that didint comply with the limit.
      :rtype df: DataFrame.
      :return columns_dropped: list of the name of the dropped columns and its nan percentage.
      :rtype columns_dropped: list[str,float].


    columns_dropped = []
    for items in nan_percentage_series.iteritems():
        if items[1] > accepted_nan_percentage:
            df = df.drop([items[0]], axis=1)
            columns_dropped.append(items)

    return df, columns_dropped


NAN_percentages = get_nan_per_col(df)
df_clean, columns_dropped = removeByNan(5, NAN_percentages, df)

print("****************************** nans drops ******************************")
print(NAN_percentages)
print(columns_dropped)


df_clean = df_clean.dropna()
df_clean = df_clean.reset_index(drop=True)
df_clean.to_csv(paths["clean_data"])

print(df_clean.info())

print("****************************** raw graphs ******************************")

#boxplot de RIESGO DE Framingham


df_clean['CALCULO DE RIESGO DE Framingham (% a 10 años)'] = df_clean['CALCULO DE RIESGO DE Framingham (% a 10 años)'].astype(int)
plt.boxplot(df_clean['CALCULO DE RIESGO DE Framingham (% a 10 años)'])
plt.title('Distribucion de riesgo de Framingham')
plt.show()


#Histograma de edad

df_clean['Edad'].hist(bins=5)
plt.title('Distribucion de la edad')
plt.xlabel('Edad')
plt.ylabel('Cuenta')
plt.show()


#bar chart género
gender_counts = df['Cod_Género'].value_counts()
ax = gender_counts.plot(kind='bar')
plt.title('Distribucion de género')
plt.xlabel('Cod_Género')
plt.ylabel('Número de personas')
ax.set_xticklabels(['H', 'M'])
plt.show()

#bar chart de fumador
smoker_counts = df['Fumador Activo'].value_counts()
ax = gender_counts.plot(kind='bar')
plt.title('Distribucion de fumadores activos')
plt.xlabel('Fumador Activo')
plt.ylabel('Número de personas')
ax.set_xticklabels(['No fuma', 'Fuma'])
plt.show()



cycle_counts = df_clean['CiclosV'].value_counts()
cycle_percentages = cycle_counts / cycle_counts.sum() * 100
fig, ax = plt.subplots()
ax.pie(cycle_counts)
ax.axis('equal')
plt.title('Distribucion ciclos de vida')
labels = [f'{i}: {j:.1f}%' for i, j in zip(cycle_percentages.index, cycle_percentages)]
plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left',labels=labels)
plt.show()



"""