import pandas as pd


def read_data(table_name, table_columns=None):
    return pd.read_csv(
        f"data\dataset\{table_name}.dat",
        delimiter="::",
        engine="python",
        header=None,
        names=table_columns,
        encoding="latin-1",
    )