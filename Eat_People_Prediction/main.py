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

    df3 = pd.DataFrame(pd.read_csv("./datasets/sample_submission.csv"))

    pred_lunch = reg.predict(df_test, mlr1, "중식계")
    df3["중식계"] = pred_lunch

    pred_dinner = reg.predict(df_test,mlr2, "석식계")
    df3["석식계"] = pred_dinner
    df3 = df3.set_index("일자")
    df3.to_csv("./result/sample_submission.csv", mode="w")

if __name__ == "__main__":
    main()