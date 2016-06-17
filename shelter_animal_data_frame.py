import pandas as pd
import patsy
from column_helper_funcs import *


class ShelterAnimalDataFrame:

    '''
    #outcome_subtype_tree = tree.DecisionTreeClassifier()
    #outcome_subtype_tree.fit(self.part_train_x[:,:-1], self.part_train_x[:,-1])
    #self.part_test_x[:,-1] = outcome_subtype_tree.predict(self.part_test_x[:,:-1])



    #self.train_data_frame[self.PREDICTED_OUTCOME_SUBTYPE] =  outcome_subtype_tree.predict(self.get_custom_x(self.get_x_cols()))
    #self.train_data_frame[self.PREDICTED_OUTCOME_SUBTYPE] = self.train_data_frame[self.PREDICTED_OUTCOME_SUBTYPE].astype(np.float64)


    ##y_mat = y.as_matrix()
        ##for i in range(len(y.columns)):
        ##    ymat[:,i] = ymat[:,i] * (i+1)

        ##ymat = np.sum(y, axis=1)


    #print(self.train_data_frame.groupby("Breed")["Breed"].count()).to_csv("data/explore_breed_counts.csv")
            #while True:
            #    breed = raw_input('Enter Breed: ')
            #    print(breed in self.train_data_frame["Breed"].values)
    '''

    def __init__(self, train_data_frame, test_data_frame, validation_data_frame):



        self.train_data_frame = train_data_frame
        self.__preprocess__(train_data_frame)

        self.test_data_frame = test_data_frame
        self.__preprocess__(test_data_frame)

        self.validation_data_frame = validation_data_frame
        self.__preprocess__(validation_data_frame)

        self.__patsify__()


    def __formula__(self):
        return None

    def __preprocess__(self, data_frame):

        if data_frame is None or data_frame.empty is True:
            return

        data_frame["DateTime"] = pd.to_datetime(data_frame["DateTime"])
        data_frame["Name"].fillna("", inplace=True)

        if "OutcomeSubtype" in data_frame.columns:
            data_frame["OutcomeSubtype"].fillna("", inplace=True)

        # mode impute
        data_frame["AgeuponOutcome"].fillna(data_frame["AgeuponOutcome"].mode().ix[0], inplace=True)

        # mode impute
        data_frame["SexuponOutcome"].fillna(data_frame["SexuponOutcome"].mode().ix[0], inplace=True)

        # split vars
        data_frame["Neutered"] = to_neutered_series(data_frame["SexuponOutcome"])
        data_frame["Sex"] = to_sex_series(data_frame["SexuponOutcome"])

        # transform age
        data_frame["AgeInDays"] = to_age_in_days_series(data_frame["AgeuponOutcome"])


        if "MaxLifeExpectancy" in data_frame.columns:
            data_frame["LifeRatio"] = to_life_ratio_series( data_frame["AgeInDays"],  data_frame["MaxLifeExpectancy"])


    def __patsify__(self):

        formula = self.formula

        all_data_frame = pd.concat([self.train_data_frame, self.test_data_frame, self.validation_data_frame])

        x = patsy.dmatrix(formula, all_data_frame, return_type='dataframe')

        train_data_row_len = self.train_data_frame.shape[0]
        test_data_row_len = self.test_data_frame.shape[0] + train_data_row_len

        self.train_x  = x.iloc[0:train_data_row_len, :]
        self.test_x = x.iloc[train_data_row_len:test_data_row_len , :]
        self.validation_x = x.iloc[test_data_row_len: , :]


        # Get Y value for all observations
        self.train_y, self.y_categories =  pd.factorize(self.train_data_frame["OutcomeType"], sort=True)

        #TODO: fix test_y factorization since test data COULD NOT HAVE ALL LABELS so we should merge the data and split x fields
        # all_y_data_frame = pd.concat([self.train_data_frame["OutcomeType], self.test_data_frame["OutcomeType"])

        if "OutcomeType" in self.test_data_frame.columns:
            self.test_y, _ = pd.factorize(self.test_data_frame["OutcomeType"], sort=True)
        else:
            self.test_y = None

        if "OutcomeType" in self.validation_data_frame.columns:
            self.validation_y, _ = pd.factorize(self.validation_data_frame["OutcomeType"], sort=True)
        else:
            self.validation_y = None

        # Get IDs for all observations

        self.train_ids = self.train_data_frame["AnimalID"]

        if "AnimalID" in self.test_data_frame.columns:
            self.test_ids = self.test_data_frame["AnimalID"]
        else:
            self.test_ids = self.test_data_frame["ID"]

        if "AnimalID" in self.validation_data_frame.columns:
            self.validation_ids = self.validation_data_frame["AnimalID"]
        elif "ID" in self.validation_data_frame.columns:
            self.validation_ids = self.validation_data_frame["ID"]
        else:
            self.validation_ids = None

    def get_models(self):
        return self.models

    def get_data(self):
        return \
            (
                (self.train_x, self.train_y, self.train_ids),
                (self.test_x, self.test_y, self.test_ids),
                (self.validation_x, self.validation_y, self.validation_ids)
            )
