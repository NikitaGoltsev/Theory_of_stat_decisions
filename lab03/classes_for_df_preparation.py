import numpy as np
import pandas as pd


def gender_parce(ln: np.int32, column: pd.DataFrame):
    dict_fot_col = {}
    uniq = column['Gender'].unique()
    for var in uniq:
        if var[0] == 'm' or var[0] == 'M':
            dict_fot_col[var] = 'M'
        else:
            dict_fot_col[var] = 'F'

    return column


class common_data():

    def __init__(self, main_data: pd.DataFrame) -> None:

        self.main_data = main_data

        return None

    def __del_nan__(self, df: pd.DataFrame) -> pd.DataFrame:

        return df

    def get_data(self) -> pd.DataFrame:

        return self.main_data
