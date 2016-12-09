import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


RF_PARAM_GRID = [{
    'n_estimators': [2000],
    'n_jobs': [-1],
    'max_features': [0.4],
    'max_depth': [20, 25, None],
}]

def main():
    df_train = pd.read_csv('./raw_data/train.csv')
    df_test = pd.read_csv('./raw_data/test.csv')

    df_train['is_test'] = False
    df_test['is_test'] = True

    df_all = pd.concat([
        df_train.drop(['Id', 'SalePrice'], axis=1),
        df_test.drop(['Id'], axis=1),
    ], axis=0)

    # some custom feature processing
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

    rgs = GridSearchCV(RandomForestRegressor(),
        param_grid=RF_PARAM_GRID,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=5,
        cv=5,
    )
    rgs.fit(x_train, y_train)

    prediction = np.expm1(rgs.best_estimator_.predict(x_test))
    pd.DataFrame({'Id': id_test, 'SalePrice': prediction}).to_csv('sub.csv', index=False)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
