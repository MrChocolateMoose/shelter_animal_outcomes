import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import normalize

class DataFramePartition:

    def __init__(self, data_file_name, initial_filter_func = None, split_pcts = [ 70.0 , 15.0 , 15.0 ], random_seed = 42):

        # read in data file
        self.data_frame = pd.read_csv(data_file_name)


        #if not self.data_frame.empty:
        #    self.data_frame =self.data_frame.tail(100)

        # filter
        if initial_filter_func is not None:
            self.data_frame = initial_filter_func(self.data_frame)

        assert(len(split_pcts) == 3)

        split_pcts = normalize(split_pcts, norm="l1").ravel()

        train_split_pct = split_pcts[0]

        if train_split_pct == 1.0:
            self.split_data_frames = None
        else:

            x_mat = self.data_frame.as_matrix()

            test_split_pct = 1 - split_pcts[0]


            x_train_mat, x_test_and_validation_mat, _, _ = \
                train_test_split(x_mat, np.zeros((x_mat.shape[0], 1)), test_size=test_split_pct, random_state = random_seed)

            split_pcts = normalize(split_pcts[1:3], norm="l1").ravel()
            validation_split_pct = 1 - split_pcts[0]

            x_test_mat, x_validation_mat, _, _ = \
                train_test_split(x_test_and_validation_mat, np.zeros((x_test_and_validation_mat.shape[0], 1)), test_size=validation_split_pct, random_state = random_seed)


            self.split_data_frames = (pd.DataFrame(x_train_mat), pd.DataFrame(x_test_mat), pd.DataFrame(x_validation_mat))

            # restore column information that was lost
            for split_data_frame in self.split_data_frames:
                split_data_frame.columns = self.data_frame.columns


    def get_data_frame(self):
        return self.data_frame

    def get_split_data_frame(self):
        return self.split_data_frames
