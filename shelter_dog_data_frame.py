from sklearn import ensemble
from sklearn import linear_model

import multiprocessing

from shelter_animal_data_frame import *
from dog_profile_data_frame import *

class ShelterDogDataFrame(ShelterAnimalDataFrame):

    def __init__(self, train_data_frame, test_data_frame, validation_data_frame):

        orig_base_formula =  """
            to_no_name_vec(Name)                +
            to_no_name_len_vec(Name)            +
            to_is_valid_name_vec(Name)          +
            to_hour_vec(DateTime)               +
            C(to_month_vec(DateTime))           +
            C(to_season_vec(DateTime))          +
            AgeInDays                           +
            C(to_bucket_vec(Color, 5.0))        +
            C(to_bucket_vec(Breed, 1.0))        +
            C(Neutered):C(Sex)
        """

        base_formula =  """
            to_no_name_vec(Name)                +
            to_no_name_len_vec(Name)            +
            to_hour_vec(DateTime)               +
            C(to_month_vec(DateTime))           +
            C(to_season_vec(DateTime))          +
            AgeInDays                           +
            C(to_bucket_vec(Color, 5.0))        +
            C(to_bucket_vec(Breed, 1.0))        +
            C(Neutered):C(Sex)
        """

        # Adding Size+LifeRatio+MaxPuppyCost 0.95107 -> 0.94778


        profile_based_formula = """
            C(BreedType) +
            Size +
            LifeRatio +
            MaxPuppyCost
        """

        orig_profile_based_formula = """
            C(BreedType) +
            Size +
            MinWeightMale +
            MaxWeightMale +
            MinHeightMale +
            MaxHeightMale +
            MinLifeExpectancy +
            MaxLifeExpectancy +
            LifeRatio +
            MinPuppyCost +
            MaxPuppyCost
        """

        #ShelterAnimalDataFrame.formula = "AgeInDays + C(Neutered):C(Sex)"

        ShelterAnimalDataFrame.formula = base_formula + " + " + profile_based_formula

        self.calculate_dog_profile(train_data_frame, "train")
        self.calculate_dog_profile(test_data_frame, "test")
        self.calculate_dog_profile(validation_data_frame, "validation")

        #train_data_frame[["Breed", "Found"]].sort(["Found"]).to_csv("data/train_found_dog_breeds.csv", index=False)
        #test_data_frame[["Breed", "Found"]].sort(["Found"]).to_csv("data/test_found_dog_breeds.csv", index=False)

        ShelterAnimalDataFrame.models = [
            ensemble.GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, min_samples_leaf=15),
            linear_model.LogisticRegression(penalty='l2', solver='lbfgs', multi_class = 'multinomial',
                                            C=1, max_iter=10000,
                                            n_jobs= multiprocessing.cpu_count())
        ]



        print(train_data_frame.isnull().sum())
        print(test_data_frame.isnull().sum())
        print(validation_data_frame.isnull().sum())


        ShelterAnimalDataFrame.__init__(self, train_data_frame, test_data_frame, validation_data_frame)


    def calculate_dog_profile(self, data_frame, data_frame_name = ""):

        if data_frame is None or data_frame.empty is True:
            return

        profile = DogProfileDataFrame()

        print("calculating " + data_frame_name + " dog breed profiles...")

        profile_data_frame = pd.concat([profile.calculate_profile_assert(breed) for breed in data_frame["Breed"].as_matrix().ravel()])
        #TODO: check sizes THIS IS WHERE THE ISSUE IS
        profile_data_frame.index = data_frame.index

        columns = [
            "Size",
            "MinWeightMale",
            "MaxWeightMale",
            "MinHeightMale",
            "MaxHeightMale",
            "MinLifeExpectancy",
            "MaxLifeExpectancy",
            "MinPuppyCost",
            "MaxPuppyCost",
            "BreedType",
            "Found"
        ]

        print(profile_data_frame.isnull().sum())
        for column in columns:
            data_frame[column] = profile_data_frame[column]
        print(data_frame.isnull().sum())
        print("finished calculating " + data_frame_name + " dog breed profiles.")
        print("")


