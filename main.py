import pandas as pd

paths = {
  "Caqueta": 'data/Caqueta_data.csv',
  "Narino": 'data/Narino_data.csv',
  "putumayo": 'data/Putumayo_data.csv'
}

df = pd.read_csv(paths["Caqueta"])

print(df.info())