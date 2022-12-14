import numpy as np


def train_test_split_by_id(data, test_size=0.2, random_seed=256):
    """
    Split the data into separate training and test data.
    :param data: Pandas dataframe to be split.
    :param test_size: The proportion of the instances to go into the test data.
    :param random_seed: Random seed to give reproducible results.
    :return: Two pandas dataframes: the training and test sets (respectively).
    """
    np.random.seed(random_seed)
    shuffled_data = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    data_train = shuffled_data[test_set_size:]
    data_test = shuffled_data[:test_set_size]
    return data.iloc[data_train], data.iloc[data_test]


def impute_feature(df, feature, strategy="median"):
    """
    Impute a dataframe to handle missing entries
    :param df: Pandas dataframe to impute.
    :param feature: String of feature name to impute.
    :param strategy: String of the strategy by which to impute. Options are: median, remove-instances, remove-feature.
    """
    if strategy == "median":
        median = np.median(df[feature])
        df[feature].fillna(median, inplace=True)
    elif strategy == "remove-instances":
        df.dropna(subset=[feature])
    elif strategy == "remove-feature":
        df.drop(feature, axis=1)


def scale_feature(df, feature, method="standardization"):
    """
    Scale a feature using a specified method
    :param df: Pandas Dataframe
    :param feature: The feature to be scaled
    :param method: The method by which to scale. Options: min-max, standardization (default)
    :return: Updated dataframe with the scaled feature
    """
    if method == "min-max":
        min_val = np.min(df[feature])
        max_val = np.max(df[feature])
        df[feature] = (df[feature] - min_val) / (max_val - min_val)

    else:
        mean = np.mean(df[feature])
        std = np.std(df[feature])
        df[feature] = (df[feature] - mean) / std

    return df
