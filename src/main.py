import pandas as pd

paths = {
  "caqueta": '../data/raw/Caqueta_data.csv',
  "narino": '../data/raw/Narino_data.csv',
  "putumayo": '../data/raw/Putumayo_data.csv'
}

common_drops = [
  "Año", "Mes", "Grupo de Riesgo", "Evento", "Afiliados","OrigenBD","DesDepto", "CodMpio", "DescMpio", "Latitud_Y_Mpio",
  "Longitud_X_Mpio","tipo_usuario", "Estado", "tipo_identifiCAcion", "Documento",	"ConCAtenar", "nombre1", "nombre2",
  "apellido1", "apellido2",	"FechaNac", "DescrCiclosV",	"QuinQ", "DescQuinQ", "Género", "EnfoqueDif",
  "Hecho Victimizante", "RUV", "Nivel_Educativo", "Ocupación","Tipo de afiliado", "Estado_Civil", "Discapacidad",
  "Grado de Discapacidad", "MUNICIPIO DONDE VIVE", "DIRECCIÓN DE DONDE VIVE", "TELEFONOS DE CONTACTO", "Zona",
  "Cód_poblado", "Nombre_poblado", "Latitud_Afiliado", "Longitud_Afiliado", "Validación_Dirección_Afiliado",
  "CodDepto_IPS", "DesDepto_IPS", "CodMpio_IPS", "DescMpio_IPS", "CodIPS", "Nombre_IPS", "Dirección_IPS", "Barrio_IPS",
  "Teléfono_IPS", "Latitud_IPS", "Longitud_IPS", "CONSUMO DE TABACO",
  "EL USUARIO CUENTA CON ATENCIÓN POR PARTE DEL EQUIPO MULTIDISCIPLINARIO DE LA SALUD ",
  "FECHA DE ENTREGA DE MEDICAMENTOS", "MODALIDAD DE ENTREGA DE MEDICAMENTOS", "FECHA INGRESO AL PROGRAMA",
  "FECHA DEL SEGUNDO CONTROL", "DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL", "FECHA DX CONFIRMADO DE HIPERTENSIÓN ARTERIAL",
  "DX CONFIRMADO DE DIABETES MELLITUS", "FECHA DX CONFIRMADO DE DIABETES MELLITUS","CLASIFICACION DIABETES",
  "FECHA DIAGNÓSTICO DISLIPIDEMIAS", "VALORACIÓN PODOLÓGICA (PIE EN DIABETICOS) POR MÉDICO GENERAL",
  "REALIZA ACTIVIDAD FISICA", "ANTECEDENTE FAMILIAR  DE ENFERMEDAD CARDIOVASCULAR", "TABAQUISMO",  "NUTRICIÓN",
  "TRABAJO SOCIAL", "MEDICINA INTERNA", "PSICOLOGIA", "NEFROLOGIA", "OFTALMOLOGÍA", "ENDOCRINOLOGIA", "CARDIOLOGIA",
  "ELECTROCARDIOGRAMA", "NEFROPROTECCIÓN", "TERAPIA SUSTITUCIÓN DIALÍTICA",
  "TALLER EDUCATIVO ENTREGA CARTILLAS", "FECHA_CLASIF_ERC", "FECHA DEL UROANALISIS", "FECHA DE ULTIMO SEGUIMIENTO ",
  "MODALIDAD COMO SE HACE EL SEGUIMIENTO DEL PACIENTE", "FECHA DE PRÓXIMO CONTROL ", "CAUSAS DE INASISTENCIA"
]

dates_drops = [
  "FECHA NUTRUCIÓN", "FECHA TRABAJO SOCIAL", "FECHA MEDICINA INTERNA", "FECHA PISCOLOGIA", "FECHA NEFROLOGIA",
  "FECHA OFTALMOMOGIA", "FECHA ENDOCRINOLOGIA", "FECHA CARDIOLOGIA", "FECHA ELECTROCARDIOGRAMA", "FECHA NEFROPROTECCIÓN"
]
caqueta_df = pd.read_csv(paths["caqueta"])
narino_df = pd.read_csv(paths["narino"])
putumayo_df = pd.read_csv(paths["putumayo"])

caqueta_df.drop(common_drops, axis=1)

narino_df.drop(common_drops, axis=1)
narino_df.drop(dates_drops, axis=1)

putumayo_df.drop(common_drops, axis=1)
putumayo_df.drop(dates_drops, axis=1)
