import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import missingno as msno


paths = {
        "caqueta": '../../data/raw_2/caqueta_data_2.csv',
        "narino": '../../data/raw_2/Narino_data_2.csv',
        "putumayo": '../../data/raw_2/Putumayo_data_2.csv',
    }

common_drops = [
        "Año", "Mes", "Programa", "Evento", "Afiliados", "OrigenBD", "DesDepto", "CodMpio", "DescMpio",
        "Latitud_Y_Mpio", "Longitud_X_Mpio", "tipo_usuario", "Estado", "tipo_identifiCAcion", "Documento", "ConCAtenar", "nombre1",
        "nombre2", "apellido1", "apellido2", "FechaNac", "CiclosV", "DescrCiclosV", "QuinQ", "DescQuinQ", "EnfoqueDif",
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
        "FECHA DE TOMA DE TGD > 150 MG/DL_DIC"
        ]


class Cleaning:

    def __init__(self, saving_path):
        global paths
        global common_drops
        global narino_putumayo_drops
        global narino_drops
        global putumayo_drops

        self.saving_path = saving_path
        self.caqueta_df = None
        self.narino_df = None
        self.putumayo_df = None
        self.first_caqueta_clean = None
        self.first_putumayo_clean = None
        self.first_narino_clean = None
        self.df = None
        self.unified_df = None
        self.df_clean = None

    def read_csv(self):
        self.caqueta_df = pd.read_csv(paths["caqueta"],low_memory=False)
        self.narino_df = pd.read_csv(paths["narino"],low_memory=False)
        self.putumayo_df = pd.read_csv(paths["putumayo"],low_memory=False)

    def drop_columns(self):
        caqueta_df_clean = self.caqueta_df.drop(common_drops, axis=1)

        narino_df_clean = self.narino_df.drop(common_drops, axis=1)
        narino_df_clean = narino_df_clean.drop(narino_drops, axis=1)
        narino_df_clean = narino_df_clean.drop(narino_putumayo_drops, axis=1)

        putumayo_df_clean = self.putumayo_df.drop(common_drops, axis=1)
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
        self.unified_df = self.df

    def replace_blanks(self):
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

    def mice_imputation(self):
        df_mice = self.df.filter(['CALCULO DE RIESGO DE Framingham (% a 10 años)',
                                  'GLICEMIA 100 MG/DL_DIC',
                                  'COLESTEROL TOTAL > 200 MG/DL_DIC',
                                  'PERIMETRO ABDOMINAL'], axis=1).copy()

        # Define MICE Imputer and fill missing values
        mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None,
                                        imputation_order='ascending')

        df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)

        self.df['CALCULO DE RIESGO DE Framingham (% a 10 años)'] = df_mice_imputed[
            'CALCULO DE RIESGO DE Framingham (% a 10 años)']
        self.df['GLICEMIA 100 MG/DL_DIC'] = df_mice_imputed['GLICEMIA 100 MG/DL_DIC']
        self.df['COLESTEROL TOTAL > 200 MG/DL_DIC'] = df_mice_imputed['COLESTEROL TOTAL > 200 MG/DL_DIC']
        self.df['PERIMETRO ABDOMINAL'] = df_mice_imputed['PERIMETRO ABDOMINAL']

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
        self.mice_imputation()
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
        return self.df
    
    def get_df_clean(self):
        return self.df_clean






"""
caqueta_df = pd.read_csv(paths["caqueta"])
narino_df = pd.read_csv(paths["narino"])
putumayo_df = pd.read_csv(paths["putumayo"])

print("*****************Caqueta*****************")
print(caqueta_df.info(verbose=True, null_counts=True))
print("*****************narino*****************")
print(narino_df.info(verbose=True, null_counts=True))
print("*****************putumayo*****************")
print(putumayo_df.info(verbose=True, null_counts=True))

caqueta_df_clean = caqueta_df.drop(common_drops, axis=1)

narino_df_clean = narino_df.drop(common_drops, axis=1)
narino_df_clean = narino_df_clean.drop(narino_drops, axis=1)
narino_df_clean = narino_df_clean.drop(narino_putumayo_drops, axis=1)

putumayo_df_clean = putumayo_df.drop(common_drops, axis=1)
putumayo_df_clean = putumayo_df_clean.drop(putumayo_drops, axis=1)
putumayo_df_clean = putumayo_df_clean.drop(narino_putumayo_drops, axis=1)

print("Csv info:")
print("*****************Caqueta*****************")
print(caqueta_df_clean.info())
print("*****************narino*****************")
print(narino_df_clean.info())
print("*****************putumayo*****************")
print(putumayo_df_clean.info())


df = pd.concat([putumayo_df_clean, narino_df_clean, caqueta_df_clean], axis=0)
print("***************** unified df")
print(df.info())

msno.matrix(df);
plt.title("Missingness - unified data frame")
plt.show()


df = df.replace(r'^\s*$', np.nan, regex=True)
df["FechaNovedadFallecido"] = df["FechaNovedadFallecido"].fillna('no aplica')
df["Coomorbilidad"] = df["Coomorbilidad"].fillna('no aplica')
df["Condición de Discapacidad"] = df["Condición de Discapacidad"].fillna('no aplica')
df["Tipo de Discapacidad"] = df["Tipo de Discapacidad"].fillna('no aplica')
df["OTROS ANTIDIABETICOS"] = df["OTROS ANTIDIABETICOS"].fillna('no aplica')
df["OTROS TRATAMIENTOS"] = df["OTROS TRATAMIENTOS"].fillna('no aplica')
df["OTROS FARMACOS ANTIHIPERTENSIVOS"] = df["OTROS FARMACOS ANTIHIPERTENSIVOS"].fillna('no aplica')

df = df.replace("#DIV/0!", np.nan)
df = df.replace("#NUM!", np.nan)
df = df.replace("#VALUE!", np.nan)


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