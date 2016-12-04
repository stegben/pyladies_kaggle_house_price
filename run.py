import pandas as pd


def main():
    df_train = pd.read_csv('./raw_data/train.csv')
    df_test = pd.read_csv('./raw_data/test.csv')

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
