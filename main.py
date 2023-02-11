import pandas as pd

def load_database(path):
    """
    Loads a csv database from a given path.

    :param path: {path of the csv file}.
    :type path: str.
    :return: A dataframe loaded with the csv information.
    :rtype: DataFrame.

    """
  df = pd.read_csv(path)
  return df