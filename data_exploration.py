import numpy as np
import pandas as pd
#import distance as dist
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import log_loss
import patsy

from column_helper_funcs import *
from data_frame_partition import *
from dog_profile_data_frame import *

def mask(df, f):
  return df[f(df)]

class FirstNameDataFrame:

    def __init__(self):
        males = pd.read_csv("data/dist.male.first", delim_whitespace=True)
        females = pd.read_csv("data/dist.female.first", delim_whitespace=True)

        self.first_name_data_frame =  pd.concat([males, females])
        #print(self.first_name_data_frame)

    def is_valid_name(self, name):
        is_valid_name = name.upper() in self.first_name_data_frame["NAME"].values

        return is_valid_name


first_name_data_frame = FirstNameDataFrame()
def to_is_valid_name_vec(name_vec):
    return name_vec.apply(lambda x: first_name_data_frame.is_valid_name(x))



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

    def __init__(self, train_data_frame, test_data_frame):



        self.train_data_frame = train_data_frame
        self.__preprocess__(train_data_frame)

        self.test_data_frame = test_data_frame
        self.__preprocess__(test_data_frame)

        self.__patsify__()


    def __formula__(self):
        return None

    def __preprocess__(self, data_frame):
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


    def __patsify__(self):

        formula = self.formula

        all_data_frame = pd.concat([self.train_data_frame, self.test_data_frame])

        x = patsy.dmatrix(formula, all_data_frame, return_type='dataframe')

        train_data_row_len = self.train_data_frame.shape[0]

        self.train_x  = x.iloc[0:train_data_row_len, :]
        self.test_x = x.iloc[train_data_row_len:, :]


        self.train_y, self.y_categories =  pd.factorize(self.train_data_frame["OutcomeType"], sort=True)


        #TODO: fix test_y factorization since test data COULD NOT HAVE ALL LABELS so we should merge the data and split x fields
        # all_y_data_frame = pd.concat([self.train_data_frame["OutcomeType], self.test_data_frame["OutcomeType"])

        if "OutcomeType" in self.test_data_frame.columns:
            self.test_y, _ = pd.factorize(self.test_data_frame["OutcomeType"], sort=True)
        else:
            self.test_y = None

        self.train_ids = self.train_data_frame["AnimalID"]

        if "AnimalID" in self.test_data_frame.columns:
            self.test_ids = self.test_data_frame["AnimalID"]
        else:
            self.test_ids = self.test_data_frame["ID"]



    def get_data(self):
        return \
            (
                (self.train_x, self.train_y, self.train_ids),
                (self.test_x, self.test_y, self.test_ids)
            )


class ShelterDogDataFrame(ShelterAnimalDataFrame):

    def __init__(self, train_data_frame, test_data_frame):

        base_formula =  """
            to_no_name_vec(Name)                +
            to_no_name_len_vec(Name)            +
            to_is_valid_name_vec(Name)          +
            to_hour_vec(DateTime)               +
            C(to_month_vec(DateTime))           +
            C(to_season_vec(DateTime))          +
            to_age_in_days_vec(AgeuponOutcome)  +
            C(to_bucket_vec(Color, 5.0))        +
            C(to_bucket_vec(Breed, 1.0))        +
            C(Neutered)                         +
            C(Sex)
        """


        profile_based_formula = """
            C(BreedType) +
            Size +
            MinWeightMale +
            MaxWeightMale +
            MinHeightMale +
            MaxHeightMale +
            MinLifeExpectancy +
            MaxLifeExpectancy +
            MinPuppyCost +
            MaxPuppyCost
        """

        ShelterAnimalDataFrame.formula = base_formula + " + " + profile_based_formula

        dog_profile_data_frame = DogProfileDataFrame()

        train_dog_profile = pd.concat([dog_profile_data_frame.calculate_profile(breed) for breed in train_data_frame["Breed"].as_matrix().ravel()])
        train_dog_profile.index = range(train_dog_profile.shape[0])

        train_data_frame["Size"] = train_dog_profile["Size"]
        train_data_frame["MinWeightMale"] = train_dog_profile["MinWeightMale"]
        train_data_frame["MaxWeightMale"] = train_dog_profile["MaxWeightMale"]
        train_data_frame["MinHeightMale"] = train_dog_profile["MinHeightMale"]
        train_data_frame["MaxHeightMale"] = train_dog_profile["MaxHeightMale"]
        train_data_frame["MinLifeExpectancy"] = train_dog_profile["MinLifeExpectancy"]
        train_data_frame["MaxLifeExpectancy"] = train_dog_profile["MaxLifeExpectancy"]
        train_data_frame["MinPuppyCost"] = train_dog_profile["MinPuppyCost"]
        train_data_frame["MaxPuppyCost"] = train_dog_profile["MaxPuppyCost"]
        train_data_frame["BreedType"] = train_dog_profile["BreedType"]
        train_data_frame["Found"] = train_dog_profile["Found"]

        test_dog_profile = pd.concat([dog_profile_data_frame.calculate_profile(breed) for breed in test_data_frame["Breed"].as_matrix().ravel()])
        test_dog_profile.index = range(test_dog_profile.shape[0])

        test_data_frame["Size"] = test_dog_profile["Size"]
        test_data_frame["MinWeightMale"] = test_dog_profile["MinWeightMale"]
        test_data_frame["MaxWeightMale"] = test_dog_profile["MaxWeightMale"]
        test_data_frame["MinHeightMale"] = test_dog_profile["MinHeightMale"]
        test_data_frame["MaxHeightMale"] = test_dog_profile["MaxHeightMale"]
        test_data_frame["MinLifeExpectancy"] = test_dog_profile["MinLifeExpectancy"]
        test_data_frame["MaxLifeExpectancy"] = test_dog_profile["MaxLifeExpectancy"]
        test_data_frame["MinPuppyCost"] = test_dog_profile["MinPuppyCost"]
        test_data_frame["MaxPuppyCost"] = test_dog_profile["MaxPuppyCost"]
        test_data_frame["BreedType"] = test_dog_profile["BreedType"]
        test_data_frame["Found"] = test_dog_profile["Found"]

        #train_data_frame[["Breed", "Found"]].sort(["Found"]).to_csv("data/train_found_dog_breeds.csv", index=False)
        #test_data_frame[["Breed", "Found"]].sort(["Found"]).to_csv("data/test_found_dog_breeds.csv", index=False)

        ShelterAnimalDataFrame.__init__(self, train_data_frame, test_data_frame)


class ShelterCatDataFrame(ShelterAnimalDataFrame):

    def __init__(self, train_data_frame, test_data_frame):
        ShelterAnimalDataFrame.formula = """
            to_no_name_vec(Name)                +
            to_no_name_len_vec(Name)            +
            to_is_valid_name_vec(Name)          +
            to_hour_vec(DateTime)               +
            C(to_month_vec(DateTime))           +
            C(to_season_vec(DateTime))          +
            to_age_in_days_vec(AgeuponOutcome)  +
            C(to_bucket_vec(Color, 5.0))        +
            C(to_bucket_vec(Breed, 1.0))        +
            C(to_is_mix_vec(Breed))             +
            C(Neutered)                         +
            C(Sex)
        """

        ShelterAnimalDataFrame.__init__(self, train_data_frame, test_data_frame)


if __name__ == '__main__':

    # 04 -16 -2016 : KAGGLE LOG LOSS: 0.77142, MEAN TEST LOG LOSS: 0.77057

    output_kaggle_predictions = False

    if output_kaggle_predictions:
        dog_train_data =  DataFramePartition("data/train.csv", filter_only_dogs).get_data_frame()
        cat_train_data =  DataFramePartition("data/train.csv", filter_only_cats).get_data_frame()

        dog_test_data  =  DataFramePartition("data/test.csv", filter_only_dogs).get_data_frame()
        cat_test_data  =  DataFramePartition("data/test.csv", filter_only_cats).get_data_frame()
    else:
        dog_train_data, dog_test_data =  \
            DataFramePartition("data/train.csv", filter_only_dogs, should_split=True).get_split_data_frame()
        cat_train_data, cat_test_data =  \
            DataFramePartition("data/train.csv", filter_only_cats, should_split=True).get_split_data_frame()

    data_sets = \
        [
            ShelterDogDataFrame(dog_train_data, dog_test_data),
            ShelterCatDataFrame(cat_train_data, cat_test_data)
        ]


    models = \
        [
            linear_model.LogisticRegression(), #solver="newton-cg", multi_class="multinomial"
            tree.DecisionTreeClassifier(),
            ensemble.bagging.BaggingClassifier(),
            ensemble.GradientBoostingClassifier()
        ]


    for model in models:

        train_mean_acc_vec = []
        test_mean_acc_vec = []
        train_log_loss_vec = []
        test_log_loss_vec = []
        submission_data_frame_vec = []


        for data_set in data_sets:

            # get partitioned data sets
            train_data, test_data = data_set.get_data()

            # get train x,y, and id
            train_data_x, train_data_y, train_data_id = train_data

            # get test x,y, and id
            test_data_x,  test_data_y, test_data_id  = test_data

            #train this data set on x and y
            model.fit(train_data_x, train_data_y)

            train_log_loss_vec.append(
                log_loss(train_data_y, model.predict_proba(train_data_x)))

            train_mean_acc_vec.append(
                model.score(train_data_x, train_data_y))

            test_prob_y = model.predict_proba(test_data_x)

            partial_submission_data_frame = pd.DataFrame(test_prob_y)
            partial_submission_data_frame.columns = data_set.y_categories

            print(len(test_data_id.values))
            print(len(partial_submission_data_frame))
            partial_submission_data_frame.insert(0, "ID", test_data_id.values)

            submission_data_frame_vec.append(partial_submission_data_frame)

            if not output_kaggle_predictions:

                test_mean_acc_vec.append(
                    model.score(test_data_x, test_data_y))

                test_log_loss_vec.append(
                    log_loss(test_data_y, test_prob_y))


        submission_data_frame = pd.concat(submission_data_frame_vec)
        submission_data_frame.sort(columns="ID", inplace=True)

        #print(submission_data_frame)


        #print(model.__class__.__name__,
        #              "Mean Train Accuracy:", train_mean_acc_vec, "Train Log Loss", train_log_loss_vec,
        #              "Mean Test Accuracy:", test_mean_acc_vec, "Test Log Loss", test_log_loss_vec)

        print(model.__class__.__name__,
              "Mean Train Accuracy:", np.mean(train_mean_acc_vec), "Train Log Loss", np.mean(train_log_loss_vec),
              "Mean Test Accuracy:", np.mean(test_mean_acc_vec), "Test Log Loss", np.mean(test_log_loss_vec))


        if output_kaggle_predictions:
            submission_data_frame.to_csv("data/" + model.__class__.__name__ + "Submission" + ".csv", index=False)
