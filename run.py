import sys

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error

# from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


if __name__ == '__main__':
    sub_fname = sys.argv[1]
    # Data preprocessing
    df_train = pd.read_csv('./raw_data/train.csv')
    df_test = pd.read_csv('./raw_data/test.csv')

    df_train['is_test'] = False
    df_test['is_test'] = True

    df_all = pd.concat([
        df_train.drop(['Id', 'SalePrice'], axis=1),
        df_test.drop(['Id'], axis=1),
    ], axis=0)

    # perform some EDA here
    # e.g. df_all['LotArea'].hist(bins=100)
    # df_all['LotArea'].hist(bins=100)
    # GrLivArea
    import ipdb; ipdb.set_trace()

    # feature engineering
    df_all['MSSubClass'] = df_all['MSSubClass'].astype(object)
    df_all['MoSold'] = df_all['MoSold'].astype(object)
    df_all['Age'] = df_all['YrSold'] - df_all['YearBuilt']
    df_all['AgeRemod'] = df_all['YrSold'] - df_all['YearRemodAdd']
    df_all['HasRemod'] = (df_all['YearBuilt'] != df_all['YearRemodAdd']).astype(int)
    df_all['HasSecondExterior'] = (df_all['Exterior1st'] != df_all['Exterior2nd']).astype(int)
    df_all['Flr1IsLarger'] = (df_all['1stFlrSF'] > df_all['2ndFlrSF']).astype(int)
    df_all['YrSold'] = df_all['YrSold'].astype(object)

    df_all_dummy = pd.get_dummies(df_all, dummy_na=True)
    df_all_feature = df_all_dummy.fillna(df_all_dummy.mean())

    df_train_feature = df_all_feature[~df_all_feature['is_test']].drop('is_test', axis=1)
    df_test_feature = df_all_feature[df_all_feature['is_test']].drop('is_test', axis=1)

    x_train = df_train_feature.values
    y_train = df_train['SalePrice'].apply(np.log1p).values
    x_test = df_test_feature.values
    id_test = df_test['Id']

    try:
        from sklearn.cross_validation import GridSearchCV
    except:
        try:
            from sklearn.model_selection import GridSearchCV
        except:
            raise("NOOOOOOOOOOOOOOOOOOOOOOOO!")

    RF_PARAM_GRID = [{
        'n_estimators': [20],
        'n_jobs': [-1],
        'max_features': [0.4, 0.2],
        'max_depth': [20, 25, None],
    }]
    rgs_cv = GridSearchCV(RandomForestRegressor(),
        param_grid=RF_PARAM_GRID,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=5,
        cv=5,
    )
    rgs_cv.fit(x_train, y_train)
    rgs = rgs_cv.best_estimator_

    # spliter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
    # for tr_idx, te_idx in spliter.split(y_train):
    #     x_subtrain = x_train[tr_idx, :]
    #     y_subtrain = y_train[tr_idx]

    #     rgs = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    #     rgs.fit(x_subtrain, y_subtrain)

    #     x_validation = x_train[te_idx, :]
    #     y_validation = y_train[te_idx]
    #     pred = rgs.predict(x_validation)

    #     pred_subtrain = rgs.predict(x_subtrain)

    #     subtrain_error = mean_squared_error(y_subtrain, pred_subtrain)
    #     validation_error = mean_squared_error(y_validation, pred)

    #     print('MSLE of subtrain: {}'.format(subtrain_error))
    #     print('MSLE of validation: {}'.format(validation_error))

    prediction = np.expm1(rgs.predict(x_test))
    pd.DataFrame({'Id': id_test, 'SalePrice': prediction}).to_csv(sub_fname, index=False)

    import ipdb; ipdb.set_trace()
