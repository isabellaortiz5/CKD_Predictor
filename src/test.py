import pandas as pd

# Example categorical dataset
data = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'green']})

# One-hot encoding
one_hot_data = pd.get_dummies(data, columns=['color'])

print(one_hot_data)