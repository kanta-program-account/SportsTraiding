import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SportsTrading:
    """_summary_
    """
    
    def __init__(self):
        # Load the data and preprocess
        self.df = pd.read_csv("./match_snapshot_public_training - training.csv").drop(columns='uuid')
        
        # Separate features and target
        self.X = self.df.drop(columns='final_delta')
        self.y = self.df['final_delta']
        
        # Metadata for easy reference
        self.feature_names = self.X.columns.tolist()
        self.target_name = 'final_delta'

        
    def Get(self) -> pd.DataFrame:
        """Retrieve the dataset as a DataFrame.

        Returns:
            pandas.core.frame.DataFrame: A DataFrame containing the feature data of the dataset.
        """
        
        # Adjust the display limit.
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        # Check for a NaN.
        # print(self.df.isnull())
        
        return self.df
    
    def Describe(self) -> pd.DataFrame:
        """
        Generate descriptive statistics for a dataset.

        Returns:
            pd.DataFrame: About basic statistics.
        """
        
        return self.df.describe()


    