import numpy as np
import pandas as pd
from missingpy import missforest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from spaceship_code.model import MainCNN

def reading_data():
    train = pd.read_csv(f'/Users/tomer/projects/Spaceship_Titanic/data/train.csv')
    test = pd.read_csv(f'/Users/tomer/projects/Spaceship_Titanic/data/train.csv')
    return train, test

def preprocessing_data(data, train = True):
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
                                         'VIP',
                                         ], drop_first = train)
    # n family members
    n_family_members = data.family_name.value_counts().rename(f'family_members_{'train' if train else 'test'}')
    data = pd.merge(n_family_members, data, left_index=True, right_on='family_name')
    data.drop([])
    x = data.drop(['Transported', 'family_name', 'Cabin', 'PassengerId', 'Name'], axis=1)
    y = data.Transported

    # Standardize x and y
    x = (x - x.mean()) / x.std()
    return x, y

def fill_missing_data(data):
    # Fill missing data using MissForest
    imputer = missforest.MissForest()
    data_imputed = imputer.fit_transform(data)
    data = pd.DataFrame(data_imputed, columns=data.columns)
    return data

def log_transform(x):
    return np.log(pd.Series([obs if obs >= np.exp(1) else None for obs in x]))

def training_model(x, y):
    # finding the best value of p using cross validation
    def create_model(p):
        return MainCNN(input_dim=x.shape[1], p=p)

    # Define the parameter grid
    param_grid = {'p': [0.1, 0.2, 0.3, 0.4, 0.5]}

    # Create a custom scorer
    def custom_scorer(model, X, y):
        model.train_new_data(X, y, epochs=10, learning_rate=0.001)
        predictions = model.predict(X)
        return -np.mean((predictions - y) ** 2)  # Negative MSE

    grid_search = GridSearchCV(estimator=create_model(0.1), param_grid=param_grid,
                               scoring=make_scorer(custom_scorer, greater_is_better=False), cv=3)
    grid_search.fit(x, y)
    best_p = grid_search.best_params_['p']
    print(f'Best p: {best_p}')

    # Train the final model with the best p
    model = create_model(best_p)
    model.train_new_data(x.to_numpy(), y.to_numpy(), epochs=10)
    return model


def predicting_new_data(model, new_data):
    return model.predict(new_data)

if __name__ == '__main__':
    train, test = reading_data()
    x_train, y_train = preprocessing_data(train)
    x_test, y_test = preprocessing_data(test)

    x_train = fill_missing_data(x_train)
    x_test = fill_missing_data(x_test)

    model = training_model(x_train, y_train)

    predictions = predicting_new_data(model, new_data)
    print(predictions)