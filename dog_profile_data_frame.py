import pandas as pd
import numpy as np
from time import sleep

class DogProfileDataFrame:

    def __init__(self):
        self.data_frame = pd.read_csv("data/dog_data.csv")
        self.__preprocess__()
        self.__create_placedholder_breed_df__()


    """
        Size

        - type is ordinal
        - replace labels with ordinal values
        - mode impute
    """
    def __size_impute__(self, to_df, from_df):
        to_df["Size"].fillna(from_df["Size"].mode().ix[0], inplace=True)

    """
        MinWeightMale, MaxWeightMale

        - type is continuous
        - stratified mean impute based on "Size"
    """
    def __male_weight_impute__(self, to_df, from_df):
        to_df["MinWeightMale"].fillna(from_df.groupby("Size")["MinWeightMale"].transform(np.mean),  inplace=True)
        to_df["MaxWeightMale"].fillna(from_df.groupby("Size")["MaxWeightMale"].transform(np.mean), inplace=True)

    """
        MinWeightFemale, MaxWeightFemale

        - type is continuous
        - impute using MinWeightMale, MaxWeightMale
    """
    def __female_weight_impute__(self, to_df, from_df):
        to_df["MinWeightFemale"].fillna(from_df["MinWeightMale"], inplace=True)
        to_df["MaxWeightFemale"].fillna(from_df["MaxWeightMale"], inplace=True)

    """
        MinHeightMale, MaxHeightMale

        - type is continuous
        - stratified mean impute based on "Size"
    """
    def __male_height_impute__(self, to_df, from_df):
        to_df["MinHeightMale"].fillna(from_df.groupby("Size")["MinHeightMale"].transform(np.mean),  inplace=True)
        to_df["MaxHeightMale"].fillna(from_df.groupby("Size")["MaxHeightMale"].transform(np.mean), inplace=True)

    """
        MinHeightFemale, MaxHeightFemale

        - type is continuous
        - impute using MinHeightMale, MaxHeightMale
    """
    def __female_height_impute__(self, to_df, from_df):
        to_df["MinHeightFemale"].fillna(from_df["MinHeightMale"], inplace=True)
        to_df["MaxHeightFemale"].fillna(from_df["MaxHeightMale"], inplace=True)

    """
        MinLifeExpectancy

        - type is continuous
        - mean impute
    """
    def __min_life_expectancy_impute__(self, to_df, from_df):
        to_df["MinLifeExpectancy"].fillna(from_df["MinLifeExpectancy"].mean(),  inplace=True)

    """
        MaxLifeExpectancy

        - type is continuous
        - impute using MinLifeExpectancy
    """
    def __max_life_expectancy_impute__(self, to_df, from_df):
        to_df["MaxLifeExpectancy"].fillna(from_df["MinLifeExpectancy"], inplace=True)

    """
        MinPuppyCost

        - type is continuous
        - mean impute
    """
    def __min_puppy_cost_impute__(self, to_df, from_df):
        to_df["MinPuppyCost"].fillna(from_df["MinPuppyCost"].mean(),  inplace=True)

    """
        MaxPuppyCost

        - type is continuous
        - impute using MinPuppyCost
    """
    def __max_puppy_cost_impute__(self, to_df, from_df):
        to_df["MaxPuppyCost"].fillna(from_df["MinPuppyCost"], inplace=True)

    def __impute__(self, to_df, from_df):
        self.__size_impute__(to_df, from_df)
        self.__male_weight_impute__(to_df, from_df)
        self.__female_weight_impute__(to_df, from_df)
        self.__male_height_impute__(to_df, from_df)
        self.__female_height_impute__(to_df, from_df)
        self.__min_life_expectancy_impute__(to_df, from_df)
        self.__max_life_expectancy_impute__(to_df, from_df)
        self.__min_puppy_cost_impute__(to_df, from_df)
        self.__max_puppy_cost_impute__(to_df, from_df)


    def __preprocess__(self):

        self.data_frame = self.data_frame[
            [
                "Breed",
                "MinWeightMale", "MinWeightFemale", "MaxWeightMale", "MaxWeightFemale", "Size",
                "MinHeightMale", "MaxHeightMale", "MinHeightFemale", "MaxHeightFemale", "MinLifeExpectancy",
                "MaxLifeExpectancy", "MinPuppyCost", "MaxPuppyCost"
            ]
        ]

        self.data_frame.replace({"Size" : {"Small" : 1, "Medium" : 2, "Large" : 3, ".*" : np.NaN}}, regex=True, inplace=True)

        self.__impute__(to_df = self.data_frame, from_df = self.data_frame)

        # GoodWithKids, CatFriendly, DogFriendly, Trainability, Shedding, Watchdog, Intelligence, Grooming,
        # Popularity,Adaptability
        # TODO

        # HypoAllergenic


        # TODO
        # create df with all dummy vars
        #groups = self.data_frame["Group"].str.get_dummies(sep=',')
        #groups.columns = "Group_" + groups.columns

    def __create_placedholder_breed_df__(self):
        placeholder_breed_row =  {
                    "MinWeightMale"  : self.data_frame["MinWeightMale"].mean(),
                    "MinWeightFemale" : self.data_frame["MinWeightFemale"].mean(),
                    "MaxWeightMale" : self.data_frame["MaxWeightMale"].mean(),
                    "MaxWeightFemale" : self.data_frame["MaxWeightFemale"].mean(),
                    "Size" : self.data_frame["Size"].mode().ix[0],
                    "MinHeightMale" : self.data_frame["MinHeightMale"].mean(),
                    "MaxHeightMale" : self.data_frame["MaxHeightMale"].mean(),
                    "MinHeightFemale" : self.data_frame["MinHeightFemale"].mean(),
                    "MaxHeightFemale" : self.data_frame["MaxHeightFemale"].mean(),
                    "MinLifeExpectancy" : self.data_frame["MinLifeExpectancy"].mean(),
                    "MaxLifeExpectancy" : self.data_frame["MaxLifeExpectancy"].mean(),
                    "MinPuppyCost" : self.data_frame["MinPuppyCost"].mean(),
                    "MaxPuppyCost" : self.data_frame["MaxPuppyCost"].mean(),
                }
        empty_breed_row =  {
                    "MinWeightMale"  : np.NaN,
                    "MinWeightFemale" : np.NaN,
                    "MaxWeightMale" : np.NaN,
                    "MaxWeightFemale" : np.NaN,
                    "Size" : np.NaN,
                    "MinHeightMale" : np.NaN,
                    "MaxHeightMale" : np.NaN,
                    "MinHeightFemale" : np.NaN,
                    "MaxHeightFemale" : np.NaN,
                    "MinLifeExpectancy" : np.NaN,
                    "MaxLifeExpectancy" : np.NaN,
                    "MinPuppyCost" : np.NaN,
                    "MaxPuppyCost" : np.NaN,
                }

        self.placeholder_breed_df = pd.DataFrame([placeholder_breed_row])
        assert(not self.placeholder_breed_df.isnull().values.any())

        self.empty_breed_df = pd.DataFrame([empty_breed_row])


    def get_data_frame(self):
        return self.data_frame

    def get_placeholder_breed_df(self):
        return self.placeholder_breed_df.copy()

    def get_empty_breed_df(self):
        return self.empty_breed_df.copy()

    '''
        Retrieve the first index in the dataframe that matches the well-known breed name. If the breed name
        cannot be found then return 'None'
    '''
    def get_df_index_from_breed_name(self, well_known_breed_name):
        matching_indices = self.data_frame[self.data_frame["Breed"] == well_known_breed_name].index.tolist()

        if len(matching_indices) == 0:
            return None
        else:
            return matching_indices[0]



    def try_get_breed_df(self, well_known_breed_name):

        #assert(well_known_breed_name is not None)

        # Fail early with a default stats if no breed was passed in
        if well_known_breed_name is None:
            return False, self.empty_breed_df

        index = self.get_df_index_from_breed_name(well_known_breed_name)

        #assert(index is not None)

        # Fail early with a default stats if breed could not be found
        if index is None:
            return False, self.empty_breed_df
        else:
            breed_df = self.data_frame.iloc[[index]].copy()

            return True, breed_df

    def flatten_breed_dfs(self, breed_dfs):

        combined_breed_dfs = pd.concat(breed_dfs)

        flattened_breed_df = self.get_empty_breed_df()
        flattened_breed_df["Size"] = combined_breed_dfs["Size"].max()
        flattened_breed_df["MinWeightMale"] = combined_breed_dfs["MinWeightMale"].mean()
        flattened_breed_df["MinWeightFemale"] = combined_breed_dfs["MinWeightFemale"].mean()
        flattened_breed_df["MaxWeightMale"] = combined_breed_dfs["MaxWeightMale"].mean()
        flattened_breed_df["MaxWeightFemale"] = combined_breed_dfs["MaxWeightFemale"].mean()
        flattened_breed_df["MinHeightMale"] = combined_breed_dfs["MinHeightMale"].mean()
        flattened_breed_df["MaxHeightMale"] = combined_breed_dfs["MaxHeightMale"].mean()
        flattened_breed_df["MinHeightFemale"] = combined_breed_dfs["MinHeightFemale"].mean()
        flattened_breed_df["MaxHeightFemale"] = combined_breed_dfs["MaxHeightFemale"].mean()
        flattened_breed_df["MinLifeExpectancy"] = combined_breed_dfs["MinLifeExpectancy"].mean()
        flattened_breed_df["MaxLifeExpectancy"] = combined_breed_dfs["MaxLifeExpectancy"].mean()
        flattened_breed_df["MinPuppyCost"] = combined_breed_dfs["MinPuppyCost"].mean()
        flattened_breed_df["MaxPuppyCost"] = combined_breed_dfs["MaxPuppyCost"].mean()
        self.__impute__(to_df = flattened_breed_df, from_df = self.data_frame)

        return flattened_breed_df


    def calculate_profile(self, breed_name):

        breed1_name, breed2_name, breed_type = None, None, "Pure"

        if "Mix" in breed_name:
            breed_name = breed_name.replace("Mix","")
            breed_type = "Mix"

        cross_breed_split = breed_name.split("/")

        breed1_name = cross_breed_split[0]
        if (len(cross_breed_split) == 2):
            breed2_name = cross_breed_split[1]
            breed_type = "Cross"

        if breed1_name is not None:
            breed1_name = breed1_name.strip()
        if breed2_name is not None:
            breed2_name = breed2_name.strip()

        has_breed1_df, breed1_df = self.try_get_breed_df(breed1_name)
        breed1_df["Breed1"] = breed1_name
        breed1_df["BreedType"] = breed_type

        has_breed2_df, breed2_df = self.try_get_breed_df(breed2_name)
        breed2_df["Breed2"] = breed2_name
        breed2_df["BreedType"] = breed_type


        if not has_breed1_df and not has_breed2_df:
            placeholder_df = self.get_placeholder_breed_df()
            placeholder_df["Breed1"] = breed1_name
            placeholder_df["Breed2"] = breed2_name
            placeholder_df["BreedType"] = breed_type
            placeholder_df["Found"] = "No"
            return placeholder_df
        elif not has_breed1_df and has_breed2_df:
            breed2_df["Found"] = "2"
            return breed2_df
        elif has_breed1_df and not has_breed2_df:
            breed1_df["Found"] = "1"
            return breed1_df
        else:
            flattened_breed_df = self.flatten_breed_dfs([breed1_df, breed2_df])
            flattened_breed_df["Breed1"] = breed1_name
            flattened_breed_df["Breed2"] = breed2_name
            flattened_breed_df["BreedType"] = breed_type
            flattened_breed_df["Found"] = "1+2"
            return flattened_breed_df
