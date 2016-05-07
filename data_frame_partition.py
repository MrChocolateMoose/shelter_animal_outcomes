import pandas as pd
from sklearn.cross_validation import train_test_split
from random import randint
import numpy as np

class DataFramePartition:

    def __init__(self, data_file_name, initial_filter_func = None, should_split = False,  split_pct = 0.25, random_seed = 42):

        # read in data file
        self.data_frame = pd.read_csv(data_file_name)

        # filter
        if initial_filter_func is not None:
            self.data_frame = initial_filter_func(self.data_frame)

        # split
        if should_split:
            data_matrix = self.data_frame.as_matrix()
            lhs_data_frame_x, rhs_data_frame_x, _, _ = \
                train_test_split(data_matrix, np.zeros((data_matrix.shape[0], 1)), test_size=split_pct, random_state = random_seed)
            self.split_data_frame = (pd.DataFrame(lhs_data_frame_x), pd.DataFrame(rhs_data_frame_x))
            self.split_data_frame[0].columns = self.data_frame.columns
            self.split_data_frame[1].columns = self.data_frame.columns

        else:
            self.split_data_frame = None


    def get_data_frame(self):
        return self.data_frame

    def get_split_data_frame(self):
        return self.split_data_frame
