from sklearn import ensemble
from sklearn import linear_model

import multiprocessing

from shelter_animal_data_frame import *

class ShelterCatDataFrame(ShelterAnimalDataFrame):

    def __init__(self, train_data_frame, test_data_frame, validation_data_frame):
        ShelterAnimalDataFrame.formula = """
            to_no_name_vec(Name)                +
            to_no_name_len_vec(Name)            +
            to_is_valid_name_vec(Name)          +
            to_hour_vec(DateTime)               +
            C(to_month_vec(DateTime))           +
            C(to_season_vec(DateTime))          +
            AgeInDays                           +
            C(to_bucket_vec(Color, 5.0))        +
            C(to_bucket_vec(Breed, 1.0))        +
            C(to_is_mix_vec(Breed))             +
            C(Neutered):C(Sex)
        """

        ShelterAnimalDataFrame.models = [
            ensemble.GradientBoostingClassifier(n_estimators=250, learning_rate=0.02, max_depth=4, min_samples_leaf=10),
            linear_model.LogisticRegression(penalty='l2', solver='lbfgs', multi_class = 'multinomial',
                                            C=1, max_iter=10000,
                                            n_jobs= multiprocessing.cpu_count())
        ]

        ShelterAnimalDataFrame.__init__(self, train_data_frame, test_data_frame, validation_data_frame)
