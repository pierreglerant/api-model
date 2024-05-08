from unittest import TestCase
import pandas as pd
import train

class TestTrain(TestCase):

    def test_clean_data(self):
        df = pd.DataFrame({
            'survived':[1,0,1],
            'pclass': [1,2,3],
            'sex':['male','female','male'],
            'age': [20,30,40]
        })

        result_df = train.clean_data(df)

        self.assertEqual(result_df.shape[0])
        self.assertEqual(result_df['sex'][0])