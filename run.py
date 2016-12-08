import pandas as pd
from sklearn.linear_model import Ridge

def main():
    df_train = pd.read_csv('./raw_data/train.csv')
    df_test = pd.read_csv('./raw_data/test.csv')

    import ipdb; ipdb.set_trace()
    x_train = df_train.drop(['Id', 'SalePrice'], axis=1).values
    y_train = df_train['SalePrice'].values
    x_test = df_test.drop(['Id'], axis=1).values
    id_test = df_test['Id']

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
