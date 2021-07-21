from utils import *
import pandas as pd
from regressor import RegressionModel

def main():
    train_data_path = "./datasets/train.csv"
    test_data_path = "./datasets/test.csv"

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'NanumGothic'
    df = pd.DataFrame(pd.read_csv(train_data_path))
    df_test = pd.DataFrame(pd.read_csv(test_data_path))

    reg = RegressionModel(df, df_test)

    mlr1, mlr2 = reg.train(df)
    reg.predict(df_test, mlr1, mlr2)

if __name__ == "__main__":
    main()