import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from word2vec import GensimWord2Vec
import matplotlib
import os
import numpy as np
import copy

matplotlib.font_manager._rebuild()

day_dict = {"월":1, "화":2, "수":3, "목":4, "금":5}
model_dict = {"lgb":LGBMRegressor(), "rfg":RandomForestRegressor(criterion="mae"),
              "xgb":GridSearchCV(XGBRegressor(objective="reg:squarederror"), {'learning_rate':[0.0, 0.1, 0.09, 0.089,0.08], 'boosting_type':['gbtree','gblinear','dart']},
                                 scoring='neg_mean_absolute_error')}

class RegressionModel:
    def __init__(self, df_train, df_test):
        self.word2vec_model_path, self.model_name = "./weight/", "word2vec.model"
        self.setWord2Vec(df_train, df_test, self.word2vec_model_path, self.model_name)

    def setWord2Vec(self, df_train, df_test, model_path, model_name):
        self.word2vec = GensimWord2Vec(emb_dim=200)
        if not os.path.isfile(model_path + model_name):  # no train model
            food_comb = self.word2vec.getFoodCombinations(df_train, df_test)
            print("traininig...")
            wv_model = self.word2vec.train(food_combinations=food_comb, model_path=self.word2vec_model_path, model_name=self.model_name)
            print("train finished!")
        else:
            wv_model = self.word2vec.load(model_path + model_name)
            print("word2vec model load finished!")
        '''word2vec end'''
        self.wv_model = wv_model

    def preProcess(self, df):
        '''Data preprocess for train_data'''
        use_data = df.copy()

        '''날짜'''
        use_data['월'] = pd.DatetimeIndex(use_data['일자']).month
        use_data['일'] = pd.DatetimeIndex(use_data['일자']).day
        use_data["요일"] = use_data["요일"].map(day_dict)
        # use_data["요일"] = LabelEncoder().fit_transform(use_data["요일"])
        # use_data["요일"] = pd.factorize(use_data["요일"])[0]
        use_data = use_data.drop(["일자"], axis=1)

        '''인원'''
        use_data["남은인원"] = use_data["본사정원수"] - (use_data["본사휴가자수"] + use_data["현본사소속재택근무자수"])
        use_data = use_data.drop(["본사휴가자수", "현본사소속재택근무자수", "본사정원수"], axis=1)

        '''메뉴'''
        use_data["조식메뉴_emb"] = use_data["조식메뉴"].apply(lambda x : self.word2vec.getEmbeddedFoodList(x, self.wv_model))
        use_data["중식메뉴_emb"] = use_data["중식메뉴"].apply(lambda x : self.word2vec.getEmbeddedFoodList(x, self.wv_model))
        use_data["석식메뉴_emb"] = use_data["석식메뉴"].apply(lambda x : self.word2vec.getEmbeddedFoodList(x, self.wv_model))
        use_data = use_data.drop(["조식메뉴", "중식메뉴", "석식메뉴"], axis=1)

        '''correlation'''
        # get_corr(use_data)

        '''normalization'''
        # use_data["본사시간외근무명령서승인건수"] = normalization(use_data["본사시간외근무명령서승인건수"])
        # use_data["중식계"] = normalization(use_data["중식계"])
        # use_data["석식계"] = normalization(use_data["석식계"])
        return use_data

    def train(self, df, method='rfg'):
        use_data = self.preProcess(df)

        # sort
        use_data = use_data[["월", "일", "요일", "남은인원", "본사출장자수", "본사시간외근무명령서승인건수", "중식메뉴_emb", "석식메뉴_emb", "중식계","석식계"]]
        # use_data.to_csv("use_data.csv", mode="w")
        x_common = use_data.iloc[:,:-4]
        emb_arr_lunch = np.round_(np.array(use_data.drop(["중식계","석식계"], axis=1).iloc[:, -2].to_numpy().tolist()), 3)
        emb_arr_dinner = np.round_(np.array(use_data.drop(["중식계","석식계"], axis=1).iloc[:, -1].to_numpy().tolist()), 3)

        '''lauch linear model'''
        train_lunch_x = x_common.to_numpy() #np.concatenate((x_common.to_numpy(), emb_arr_lunch), axis=1)
        train_lunch_y = use_data["중식계"]
        mlr_lunch = model_dict[method]
        mlr_lunch.fit(train_lunch_x, train_lunch_y)
        lunch_out = copy.deepcopy(mlr_lunch)

        train_dinner_x = x_common.to_numpy() #np.concatenate((x_common.to_numpy(), emb_arr_dinner), axis=1)
        train_dinner_y = use_data["석식계"]
        mlr_dinner = model_dict[method]
        mlr_dinner.fit(train_dinner_x, train_dinner_y)
        dinner_out = copy.deepcopy(mlr_dinner)

        if method =="xgb":
            lunch_out = lunch_out.best_estimator_
            dinner_out = dinner_out.best_estimator_
        return lunch_out, dinner_out

    def predict(self, df_test, mlr, mode="중식계"):
        '''Data preprocess for test_data'''
        use_data2 = self.preProcess(df_test)

        # sort
        use_data2 = use_data2[["월", "일", "요일", "남은인원", "본사출장자수", "본사시간외근무명령서승인건수", "중식메뉴_emb", "석식메뉴_emb"]]

        x_common = use_data2.iloc[:,:-2]
        emb_arr_lunch = np.array(use_data2.iloc[:, -2].to_numpy().tolist())
        emb_arr_dinner = np.array(use_data2.iloc[:, -1].to_numpy().tolist())

        '''lauch linear model'''
        test_lunch_x = x_common.to_numpy() #np.concatenate((x_common.to_numpy(), emb_arr_lunch), axis=1)
        test_dinner_x = x_common.to_numpy() #np.concatenate((x_common.to_numpy(), emb_arr_dinner), axis=1)

        if mode=="중식계":
            predicted = mlr.predict(test_lunch_x)

        else:
            predicted = mlr.predict(test_dinner_x)
        print(predicted)
        return predicted


    # def predict(self, df_test, mlr, mlr2):
    #     '''Data preprocess for test_data'''
    #     use_data2 = self.preProcess(df_test)
    #     x_common = use_data2.iloc[:,:-3]
    #     emb_arr_lunch = np.array(use_data2.iloc[:, -2].to_numpy().tolist())
    #     emb_arr_dinner = np.array(use_data2.iloc[:, -1].to_numpy().tolist())
    #
    #     '''lauch linear model'''
    #     test_launch_x = x_common.to_numpy() #np.concatenate((x_common.to_numpy(), emb_arr_lunch), axis=1)
    #     test_dinner_x = x_common.to_numpy() #np.concatenate((x_common.to_numpy(), emb_arr_dinner), axis=1)
    #
    #     # launch
    #     predict_lunch = mlr.predict(test_launch_x)
    #     predict_dinner = mlr2.predict(test_dinner_x)
    #     print(predict_lunch[0])
    #     print(predict_dinner[0])
    #     df3 = pd.DataFrame(pd.read_csv("./datasets/sample_submission.csv"))
    #     df3["중식계"] = predict_lunch
    #     df3["석식계"] = predict_dinner
    #     df3 = df3.set_index("일자")
    #     df3.to_csv("./result/sample_submission.csv", mode="w")


