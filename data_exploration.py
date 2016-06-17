import numpy as np
import pandas as pd
#import distance as dist
from sklearn import linear_model
from sklearn import tree
from sklearn.grid_search import GridSearchCV
import patsy
import multiprocessing


import os
import sys
import timeit


from data_frame_partition import *
from shelter_cat_data_frame import *
from shelter_dog_data_frame import *
from run_sklearn_models import *


def mask(df, f):
  return df[f(df)]

def create_optimized_sklearn_log_reg_model(data_sets):

    param_grid = {
        # 'fit_intercept': [ True, False],
        'C' : [1, 10, 100, 1000], #0.001, 0.01, 0.1,
        'x' : [100, 1000, 10000]
        # 'tol' : [] tol=0.0001
     }

    model = linear_model.LogisticRegression(penalty='l2', solver='lbfgs', multi_class = 'multinomial', n_jobs= multiprocessing.cpu_count()/2 )

    linear_model.LogisticRegression(penalty='l2', solver='lbfgs', multi_class = 'multinomial',
                                    C=1, max_iter=10000,
                                    n_jobs= multiprocessing.cpu_count())



    create_optimized_sklearn_model(data_sets, model, param_grid)


def create_optimized_sklearn_boosting_model(data_sets):

    param_grid = {'learning_rate': [ 0.05, 0.02, 0.01],
                  'max_depth': [3, 4, 5, 6],
                  'min_samples_leaf': [5, 10, 15, 25],
                  'n_estimators' : [100, 250]
                  }

    model = ensemble.GradientBoostingClassifier()

    create_optimized_sklearn_boosting_model(data_sets, model, param_grid)


def create_optimized_sklearn_model(data_sets, model, param_grid):

    for data_set in data_sets:

        print("Optimizing the following model (" + type(model).__name__ + ") with the following data (" + type(data_set).__name__ + ")")

        # get partitioned data sets
        train_data, test_data, validation_data = data_set.get_data()

        # get train x,y, and id
        train_data_x, train_data_y, train_data_id = train_data

        # get test x,y, and id
        test_data_x, test_data_y, test_data_id = test_data

        # get validation x,y, and id
        validation_data_x, validation_data_y, validation_data_id = validation_data

        # combine all data
        data_x = np.vstack((train_data_x, test_data_x, validation_data_x))
        data_y = np.hstack((train_data_y, test_data_y, validation_data_y))

        gs_cv = GridSearchCV(model, param_grid, n_jobs=multiprocessing.cpu_count()).fit(data_x, data_y)

        # best hyperparameter setting
        print(gs_cv.best_params_)


if __name__ == '__main__':

    #quit()

    # 04 -16 -2016 : KAGGLE LOG LOSS: 0.77142, MEAN TEST LOG LOSS: 0.77057

    #('Mean Train Accuracy:', 0.71537518143637557, 'Train Log Loss', 0.69327796810227982, 'Mean Test Accuracy:', 0.68882885883092948, 'Test Log Loss', 0.75743851218433)
    #('Mean Train Accuracy:', 0.71537518143637557, 'Train Log Loss', 0.69327796810227982, 'Mean Test Accuracy:', 0.68882885883092948, 'Test Log Loss', 0.75743895083263824)
    output_kaggle_predictions = True

    if output_kaggle_predictions:

        split_pcts = [1.0, 0.0, 0.0]

        dog_train_data =  DataFramePartition("data/train.csv", filter_only_dogs, split_pcts = split_pcts).get_data_frame()
        cat_train_data =  DataFramePartition("data/train.csv", filter_only_cats, split_pcts = split_pcts).get_data_frame()

        dog_test_data  =  DataFramePartition("data/test.csv", filter_only_dogs, split_pcts = split_pcts).get_data_frame()
        cat_test_data  =  DataFramePartition("data/test.csv", filter_only_cats, split_pcts = split_pcts).get_data_frame()

        dog_validation_data = pd.DataFrame()
        cat_validation_data = pd.DataFrame()
    else:

        split_pcts = [0.70, 0.30, 0.0]

        dog_train_data, dog_test_data, dog_validation_data =  \
            DataFramePartition("data/train.csv", filter_only_dogs, split_pcts = split_pcts).get_split_data_frame()
        cat_train_data, cat_test_data, cat_validation_data =  \
            DataFramePartition("data/train.csv", filter_only_cats, split_pcts = split_pcts).get_split_data_frame()

    data_sets = \
        [
            ShelterDogDataFrame(dog_train_data, dog_test_data, dog_validation_data),
            ShelterCatDataFrame(cat_train_data, cat_test_data, cat_validation_data)
        ]

    #create_optimized_sklearn_log_reg_model(data_sets)
    #create_optimized_sklearn_boosting_model(data_sets)
    #run_dbn_model(data_sets[0])
    run_sklearn_models(data_sets, output_kaggle_predictions)
