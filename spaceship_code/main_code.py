import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from spaceship_code.model import MainCNN
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

def reading_data():
    train = pd.read_csv(f'/Users/tomer/projects/Spaceship_Titanic/data/train.csv')
    test = pd.read_csv(f'/Users/tomer/projects/Spaceship_Titanic/data/test.csv')
    return train, test

def preprocessing_data(train_data: pd.DataFrame,
                       test_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data, test_data = add_remove_variables(train_data), add_remove_variables(test_data)
    train_data, test_data = add_family_members_in_both(train_data, test_data)
    train_data, test_data = fill_missing_data_smart(train_data, test_data)
    train_data, test_data = get_dummies(train_data, True), get_dummies(test_data, False)
    test_data = test_data.loc[:, train_data.columns]
    train_data, test_data = standardize(train_data), standardize(test_data)
    return train_data, test_data

def add_remove_variables(data):
    data = add_names_data(data)
    data['cabin_first_letter'] = data.Cabin.str.split('/').str.get(0)
    data['cabin_number'] = data.Cabin.str.split('/').str.get(1).astype(float)
    data['cabin_second_letter'] = data.Cabin.str.split('/').str.get(2)

    data['family_name'] = data.Name.str.split(' ').str.get(1)
    data['starting_letter_in_first_name'] = data.Name.str.get(0)
    data['ending_letter_in_first_name'] = data.Name.str.get(-1)
    data['starting_letter_in_last_name'] = data.Name.str.get(0)
    data['ending_letter_in_last_name'] = data.Name.str.get(-1)
    # log transforming some variables
    data['RoomService_log'] = log_transform(data.RoomService)
    data['FoodCourt_log'] = log_transform(data.FoodCourt)
    data['ShoppingMall_log'] = log_transform(data.ShoppingMall)
    data['VRDeck_log'] = log_transform(data.VRDeck)

    n_family_members = data.family_name.value_counts().rename('family_members')
    data = pd.merge(n_family_members, data, left_index=True, right_on='family_name', how='right')

    data = data.drop(['Cabin', 'PassengerId', 'Name'], axis=1)
    return data


def add_family_members_in_both(df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Add a column 'family_members_in_both' to both DataFrames with the sum of 'family_members'
    from both DataFrames grouped by 'family_name'.

    :param df1: First DataFrame with 'family_name' and 'family_members'.
    :param df2: Second DataFrame with 'family_name' and 'family_members'.
    :return: Two updated DataFrames with the new column added.
    """
    # Combine family_members from both DataFrames
    combined = pd.concat([df1[['family_name', 'family_members']],
                          df2[['family_name', 'family_members']]])

    # Calculate total family_members by family_name
    family_members_sum = combined.groupby('family_name')['family_members'].sum().reset_index()
    family_members_sum.rename(columns={'family_members': 'family_members_in_both'}, inplace=True)

    # Merge the total family members back into each DataFrame
    df1 = df1.merge(family_members_sum, on='family_name', how='left')
    df2 = df2.merge(family_members_sum, on='family_name', how='left')
    df1.drop('family_name', axis=1 ,inplace=True)
    df2.drop('family_name', axis=1 ,inplace=True)

    return df1, df2
def get_dummies(data, is_train):
    # onehot encoding for categorical variables
    data = pd.get_dummies(data, columns=['cabin_first_letter',
                                         'cabin_second_letter',
                                         'starting_letter_in_first_name',
                                         'ending_letter_in_first_name',
                                         'starting_letter_in_last_name',
                                         'ending_letter_in_last_name',
                                         'HomePlanet',
                                         'CryoSleep',
                                         'Destination',
                                         'likelyGender',
                                         'VIP',
                                         ], drop_first=is_train)
    return data


def standardize(data):

    # Standardize x and y
    non_binary_columns = data.columns[data.nunique() > 2]
    data[non_binary_columns] = ((data[non_binary_columns] - data[non_binary_columns].mean())
                                / data[non_binary_columns].std())
    return data


def fill_missing_data_smart(data_train: pd.DataFrame,
                            data_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Fill missing data using IterativeImputer for both numerical and categorical variables.
    Add one-hot encoded columns for imputed observations.

    :param data_train: pd.DataFrame, input data with missing values.
    :param data_test: pd.DataFrame, input data with missing values.

    :return: pd.DataFrame, data with imputed values and one-hot encoding for imputed observations.
    """
    # Separate numerical and categorical columns
    numerical_cols = data_train.select_dtypes(include=['number']).columns
    categorical_cols = data_train.select_dtypes(include=['object', 'category']).columns


    # define imputers
    numerical_imputer = IterativeImputer(max_iter=10, random_state=42)
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Identify missing entries
    missing_mask_train = data_train.isna()
    missing_mask_test = data_test.isna()
    data_train_imputed = data_train.copy()
    data_test_imputed = data_test.copy()

    # impute categorical data
    data_train_imputed[categorical_cols] = categorical_imputer.fit_transform(data_train[categorical_cols])
    data_test_imputed[categorical_cols] = categorical_imputer.transform(data_test[categorical_cols])
    # Perform imputation
    data_train_imputed[numerical_cols] = numerical_imputer.fit_transform(data_train[numerical_cols])
    data_test_imputed[numerical_cols] = numerical_imputer.transform(data_test[numerical_cols])
    # Restore data types
    data_train_imputed[numerical_cols] = data_train_imputed[numerical_cols].astype(float)
    data_train_imputed[categorical_cols] = data_train_imputed[categorical_cols].astype(str)
    data_test_imputed[numerical_cols] = data_test_imputed[numerical_cols].astype(float)
    data_test_imputed[categorical_cols] = data_test_imputed[categorical_cols].astype(str)

    # Add one-hot encoded columns for imputed observations
    for column in data_train.columns:
        onehot_col = f"{column}_imputed"
        data_train_imputed[onehot_col] = missing_mask_train[column].astype(int)
        data_test_imputed[onehot_col] = missing_mask_test[column].astype(int)

    return data_train_imputed, data_test_imputed

def log_transform(x):
    return pd.Series([None if obs is None else (0 if obs < np.exp(1) else np.log(obs)) for obs in x])
def training_models(x, y) -> (MainCNN, MainCNN, MainCNN):
    # finding the best value of p using cross validation
    lr = 0.001
    patience = 50
    delta = 0.00005
    dropout_values = [0.25, 0.3, 0.35, 0.4]
    weights = [1e-4, 1e-3, 1e-2]
    epochs = 20000
    best_params = cross_validate_dropout(x=x,
                           y=y,
                           dropout_values=dropout_values,
                           num_folds=4,
                           epochs=epochs,
                           learning_rate=lr,
                           early_stopping_patience=patience,
                           early_stopping_min_delta=delta,
                           weights = weights
                           )
    models = []
    for i in range(3):
        cnn_model = MainCNN(x.shape[1], p=best_params[i])
        cnn_model.train_new_data(x.to_numpy().astype(float), y.to_numpy(),
                             epochs=20000,
                             learning_rate=lr,
                             early_stopping_patience= patience,
                             early_stopping_min_delta=delta,
                             print_every_x= 50,
                             w_decay= best_params[3+i])
        models.append(cnn_model)
    return models

def predicting_new_data(model, new_data):
    """
    This function is used to predict
    :param model:
    :param new_data:
    :return:
    """
    return model.predict(new_data.to_numpy().astype(float))


def cross_validate_dropout(x: np.array,
                           y: np.array,
                           dropout_values: list,
                           num_folds: int = 5,
                           epochs: int = 50,
                           learning_rate: float = 0.005,
                           early_stopping_patience: int = 10,
                           early_stopping_min_delta: float = 0.0,
                           weights = [1e-4]) -> (float, float, float, float, float ,float):
    """
    Perform K-Fold Cross-Validation to find the best dropout proportion.
    :param x: np.array, the input data.
    :param y: np.array, the outcome data.
    :param dropout_values: list, different dropout proportions to test.
    :param num_folds: int, the number of folds for cross-validation.
    :param epochs: int, the number of desired epochs for each fold.
    :param learning_rate: float, the learning rate for the optimizer.
    :param early_stopping_patience: int, patience for early stopping.
    :param early_stopping_min_delta: float, minimum improvement for early stopping.
    :param weights: iterable, iterable object of weight for the weight_decay parameter.
    :return: float, the best dropout proportion.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_p = None
    second_best_p = None
    third_best_p = None
    best_w = None
    second_best_w = None
    third_best_w = None

    best_avg_loss = float('inf')
    second_best_avg_loss = float('inf')
    third_best_avg_loss = float('inf')

    x, y = x.reset_index().drop('index', axis=1), y.reset_index().drop('index', axis=1)
    # Iterate over each dropout proportion
    for p in dropout_values:
        for w_decay in weights:
            print(f'starting dropout {p}, weight {w_decay}')
            fold_losses = []

            # Perform K-Fold Cross-Validation
            for train_index, val_index in kf.split(x):
                x_train, x_val = x.loc[train_index], x.loc[val_index]
                y_train, y_val = y.loc[train_index], y.loc[val_index]

                # Initialize the model
                model = MainCNN(input_dim=x.shape[1], p=p)
                model.train_new_data(x_train.to_numpy().astype(float),
                                     y_train.astype(int).to_numpy(),
                                     epochs,
                                     learning_rate,
                                     early_stopping_patience,
                                     early_stopping_min_delta,
                                     print_every_x=100,
                                     w_decay=w_decay)

                # Validate on the validation set
                y_val_pred = model.predict(x_val.to_numpy().astype(float))
                val_loss = 1-np.sum(y_val.to_numpy().flatten() == (y_val_pred.flatten()>0.5))/y_val.shape[0]  # Mean Squared Error for validation
                fold_losses.append(val_loss)

            # Average validation loss for this dropout value
            avg_loss = np.mean(fold_losses)
            print(f"Dropout {p}, weight_decay {w_decay}: Avg Validation Loss = {avg_loss:.4f}")

            # Track the best dropout proportion
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                third_best_p = second_best_w
                third_best_w = second_best_w
                second_best_p = best_p
                second_best_w = best_w
                best_p = p
                best_w = w_decay
            else:
                if second_best_avg_loss > avg_loss:
                    third_best_w = second_best_w
                    third_best_p = second_best_p
                    second_best_w = w_decay
                    second_best_p = p
                else:
                    if third_best_avg_loss > avg_loss:
                        third_best_w = w_decay
                        third_best_p = p


    print(f"Best Dropout Proportion: {best_p} with Avg Validation Loss: {best_avg_loss:.4f}")
    return best_p, second_best_p, third_best_p, best_w, second_best_w, third_best_w

def add_names_data(df):
    names = pd.read_csv('/Users/tomer/projects/Spaceship_Titanic/data/namsor_genderize-name_first_names.csv')
    names = names.loc[:, ['firstName', 'likelyGender', 'genderScale', 'score', 'probabilityCalibrated']]
    names = names.set_index('firstName')
    df['firstName'] = df.Name.str.split(' ').str.get(0)
    df = pd.merge(df,
                  names.loc[[name for name in df.firstName]], left_on='firstName', right_index=True).drop_duplicates()
    return df.drop('firstName', axis=1)

if __name__ == '__main__':
    print('reading_data')
    train, test = reading_data()
    # x_train = train.drop('Transported', axis=1)
    # y_train = train.Transported
    # print('preprocessing data')
    # x_train, x_test = preprocessing_data(x_train, test)
    # print('saving pickle')
    # x_train.to_pickle('/Users/tomer/projects/Spaceship_Titanic/data/x_train_cleaned.pkl')
    # x_test.to_pickle('/Users/tomer/projects/Spaceship_Titanic/data/x_test_cleaned.pkl')
    # y_train.to_pickle('/Users/tomer/projects/Spaceship_Titanic/data/y_train_cleaned.pkl')

    #print('read cleaned data')
    x_train = pd.read_pickle('/Users/tomer/projects/Spaceship_Titanic/data/x_train_cleaned.pkl')
    x_test = pd.read_pickle('/Users/tomer/projects/Spaceship_Titanic/data/x_test_cleaned.pkl')
    y_train= pd.read_pickle('/Users/tomer/projects/Spaceship_Titanic/data/y_train_cleaned.pkl')
    print('training model')
    models = training_models(x_train, y_train)
    predictions = ((pd.concat([pd.Series(predicting_new_data(model, x_test)[:,0]) for model in models],
                             axis=1)).mean(axis=1))>0.5
    test_res = pd.concat([test.PassengerId, predictions.rename('Transported')], axis=1)
    test_res.to_csv('/Users/tomer/projects/Spaceship_Titanic/data/results_with_names_data_mean.csv' ,index = False)
    print(predictions)
