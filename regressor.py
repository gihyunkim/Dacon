import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from word2vec import GensimWord2Vec
import matplotlib
import os

matplotlib.font_manager._rebuild()

class RegressionModel:
    def __init__(self, df, df_test):
        self.df_train = df
        self.df_test = df_test
        self.word2vec_model_path, self.model_name = "./weight/", "word2vec.model"
        self.setWord2Vec(self.word2vec_model_path, self.model_name)

    def setWord2Vec(self, model_path, model_name):
        self.word2vec = GensimWord2Vec(emb_dim=200)
        if not os.path.isfile(model_path + model_name):  # no train model
            food_comb = self.word2vec.getFoodCombinations(self.df_train, self.df_test)
            print("traininig...")
            wv_model = self.word2vec.train(food_combinations=food_comb, model_path=self.word2vec_model_path, model_name=self.model_name)
            print("train finished!")
        else:
            wv_model = self.word2vec.load(model_path + model_name)
            print("word2vec model load finished!")
        '''word2vec end'''
        self.wv_model = wv_model

    def preProcess(self, use_data):
        '''Data preprocess for train_data'''

        '''날짜'''
        use_data['월'] = pd.DatetimeIndex(use_data['일자']).month
        use_data['일'] = pd.DatetimeIndex(use_data['일자']).day
        use_data["요일"] = pd.factorize(use_data["요일"])[0]

        '''인원'''
        use_data["남은인원"] = use_data["본사정원수"] - (use_data["본사휴가자수"] + use_data["현본사소속재택근무자수"])
        use_data = use_data.drop(["본사휴가자수", "현본사소속재택근무자수"], axis=1)

        '''메뉴'''
        use_data["조식메뉴_emb"] = use_data["조식메뉴"].apply(lambda x : self.word2vec.getEmbeddedFoodList(x, self.wv_model))
        use_data["중식메뉴_emb"] = use_data["중식메뉴"].apply(lambda x : self.word2vec.getEmbeddedFoodList(x, self.wv_model))
        use_data["석식메뉴_emb"] = use_data["석식메뉴"].apply(lambda x : self.word2vec.getEmbeddedFoodList(x, self.wv_model))
        use_data = use_data.drop(["조식메뉴", "중식메뉴", "석식메뉴"], axis=1)
        print(use_data)
        exit(-1)

        '''correlation'''
        # get_corr(use_data)

        '''normalization'''
        # use_data["본사시간외근무명령서승인건수"] = normalization(use_data["본사시간외근무명령서승인건수"])
        # use_data["중식계"] = normalization(use_data["중식계"])
        # use_data["석식계"] = normalization(use_data["석식계"])
        return use_data

    def train(self, df):
        use_data = self.preProcess(df)

        '''lauch linear model'''
        train_launch_x = use_data[["요일", "남은인원", "본사출장자수", "본사시간외근무명령서승인건수"]]
        train_launch_y = use_data[["중식계"]]
        mlr_launch = RandomForestRegressor(criterion="mae")
        mlr_launch.fit(train_launch_x, train_launch_y)

        train_dinner_x = use_data[["요일", "남은인원", "본사출장자수", "본사시간외근무명령서승인건수"]]
        train_dinner_y = use_data[["석식계"]]
        mlr_dinner = RandomForestRegressor(criterion="mae")
        mlr_dinner.fit(train_dinner_x, train_dinner_y)

        return mlr_launch, mlr_dinner

    def predict(self, df_test, mlr, mlr2):
        '''Data preprocess for test_data'''
        use_data2 = self.preProcess(df_test)

        test_x = use_data2[["요일", "남은인원", "본사출장자수", "본사시간외근무명령서승인건수"]]

        # launch
        predict_launch = mlr.predict(test_x)
        predict_dinner = mlr2.predict(test_x)
        df3 = pd.DataFrame(pd.read_csv("./datasets/sample_submission.csv"))
        df3["중식계"] = predict_launch
        df3["석식계"] = predict_dinner
        df3 = df3.set_index("일자")
        df3.to_csv("./result/sample_submission.csv", mode="w")


