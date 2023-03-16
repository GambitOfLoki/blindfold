import pandas as pd
import numpy as np

def train_test_split(df, group_name, label_name, test_percentage=0.2, random=True):
    """
    Parameters
    ----------
    df: pandas.DataFrame, dataframe containing you modelling dataset
    group_name: string, name of the column that is unique for a every list of items
    label_name: string, target variable
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: boolean, optional
        
    Returns
    -------
    X_train (dataframe), y_train(series), qids_train(array), X_test(dataframe), y_test(series), qids_test(array)
    """

    unique_groups = list(df[group_name].unique())
    
    # if we want randomness in split
    if random:
        np.random.shuffle(unique_groups)

    training = unique_groups[:int(len(unique_groups)*(1-test_percentage))] 
    testing = unique_groups[int(len(unique_groups)*test_percentage):]

    df_train = df[df[group_name].isin(training)]
    df_test = df[df[group_name].isin(testing)]
    
    # Creating a numpy array which contains train_group
    qids_train = df_train.groupby(group_name)[group_name].count().to_numpy()
    # Keeping only the features on which we would train our model 
    X_train = df_train.drop([group_name, label_name], axis = 1)
    # Relevance label for train
    y_train = df_train[label_name].astype(int)

    # Creating a numpy array which contains val_group
    qids_test = df_test.groupby(group_name)[group_name].count().to_numpy()
    # Keeping only the features on which we would test our model
    X_test = df_test.drop([group_name, label_name], axis = 1)
    # Relevance label for test
    y_test = df_test[label_name].astype(int)
    
    return X_train, y_train, qids_train, X_test, y_test, qids_test, training, testing