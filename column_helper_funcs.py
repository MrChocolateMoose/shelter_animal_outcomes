def filter_only_dogs(data_frame):
    return data_frame.loc[data_frame["AnimalType"] == "Dog"]

def filter_only_cats(data_frame):
    return data_frame.loc[data_frame["AnimalType"] == "Cat"]


def to_no_name_vec(name_vec):
    return name_vec.apply(lambda x: not x)

def to_no_name_len_vec(name_vec):
    return name_vec.apply(lambda x: len(x))


def to_bucket_vec ( vec, threshold_pct = 5.0 ):
    val_counts = vec.value_counts()

    max = val_counts.max()
    clipped_vec = val_counts.apply(lambda x, max=max, threshold_pct=threshold_pct: clip_below_threshold(x, max, threshold_pct))

    return vec.apply(lambda x, clipped_vec=clipped_vec: to_bucket(x, clipped_vec) )

def to_bucket ( color, clipped_vec ):
    if color in clipped_vec.index.values:
        if clipped_vec[color] == -1:
            return "Other"
        else:
            return color
    else:
        return "Other"

def to_is_mix_vec (breed_vec):
    return breed_vec.apply(to_is_mixed_breed)

def to_is_mixed_breed (breed_name):
    if type(breed_name) is str:
        return 1 #"/" in breed_name or "Mix" in breed_name
    else:
        return 0

def to_col_type_name (clipped_val, col_name, col_type):
   if clipped_val > 0:
       return col_type + ": " + col_name
   else:
       return col_type + ": Other"

def clip_below_threshold (val, max, pct):
    if val >= (pct * max / 100.0):
        return val
    else:
        return -1

def to_age_in_days_vec ( age_vec ):
    return age_vec.apply(to_age_in_days)

def to_age_in_days ( age_str ):
    # non-string or empty str
    if not age_str or type(age_str) is not str:
        return -1

    age_number, age_unit = tuple(age_str.split())

    if age_number.isdigit():
        age_number = int(age_number)
    else:
        return -1

    if "year" in age_unit:
        return age_number * 365.25
    elif "month" in age_unit:
        return age_number * (365.25) / 12
    elif "day" in age_unit:
        return age_number
    elif "week" in age_unit:
        return age_number * (365.25) / 52
    else:
        return -1

def to_month_vec ( date_time_vec ):
    return date_time_vec.apply(lambda x: x.month)

def to_hour_vec ( date_time_vec ):
    return date_time_vec.apply(lambda x: x.hour)


def to_season_vec ( date_time_vec ):
    return date_time_vec.apply(to_season)

def to_season( date_time ):
    if  (date_time.month >= 3 and date_time.day >= 1) and \
        (date_time.month <= 5 and date_time.day <= 31):
        return 1
    elif (date_time.month >= 6 and date_time.day >= 1) and \
         (date_time.month <= 8 and date_time.day <= 31):
        return 2
    elif    (date_time.month >= 9 and date_time.day >= 1) and\
            (date_time.month <= 11 and date_time.day <= 30):
        return 3
    else:
        return 4

