import pandas as pd

paths = {
  "Caqueta": 'src/data/Caqueta_data.csv',
  "Narino": 'src/data/Narino_data.csv',
  "putumayo": 'src/data/Putumayo_data.csv'
}

df = pd.read_csv(paths["Caqueta"])
