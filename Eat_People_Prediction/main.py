import pandas as pd
from regressor import RegressionModel
import numpy as np

def main():
    train_data_path = "./datasets/train.csv"
    test_data_path = "./datasets/test.csv"

    '''Set Dataframe'''
    df = pd.DataFrame(pd.read_csv(train_data_path))
    df_test = pd.DataFrame(pd.read_csv(test_data_path))

    reg = RegressionModel(df, df_test)

    '''train'''
    mlr_lunch, mlr_dinner = reg.train(df, method="rfg")

    '''test'''
    pred_lunch, pred_dinner = reg.predict(df_test, mlr_lunch, mlr_dinner)

    '''save result'''
    reg.save_csv(pred_lunch, pred_dinner)

if __name__ == "__main__":
    main()