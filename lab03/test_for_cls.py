import unittest
from classes_for_df_preparation import common_data, gender_parce

import pandas as pd


class full_test(unittest.TestCase):

    def setUp(self) -> None:
        df_nul = pd.DataFrame()
        self.class_of_df = common_data(df_nul)
        return super().setUp()
    '''
    def null_setup(self) -> None:

       return None
    '''

    def test_par(self):
        df = pd.DataFrame(['Male'])
        df.columns = ['Gender']

        df_t = pd.DataFrame(['M'])
        df_t.columns = ['Gender']
        self.assertNotEqual(gender_parce(1, df), df_t)


if __name__ == "__main__":
    unittest.main()
