import pandas as pd
from gensim.models import Word2Vec
import os
import numpy as np

'''word2vec start'''
class GensimWord2Vec:
    def __init__(self, emb_dim=200):
        self.emb_dim = emb_dim

    def getFoodCombinations(self, df, df_test):
        menu_df = pd.concat([df[["조식메뉴", "중식메뉴", "석식메뉴"]], df_test[["조식메뉴", "중식메뉴", "석식메뉴"]]])
        food_combinations = []
        for i in ['조식메뉴', '중식메뉴', '석식메뉴']:
            food_combinations += menu_df[i].apply(self.split_process).to_list()
        return food_combinations

    def split_process(self, x):
        x_ = []
        x = x.split(' ')
        for i in x:
            if '(' in i and ':' in i and ')' in i:
                continue
            if '/' in i:
                x_.extend(i.split('/'))
            else:
                x_.append(i)
        x_ = list(set(x_))
        x_.remove('')
        return x_

    def train(self, food_combinations, emb_dim=200, epochs=5000, model_path="./model/", model_name="word2vec.model"):
        model = Word2Vec(sentences=food_combinations, vector_size=emb_dim, window=7, min_count=0, workers=4, sg=0, epochs=epochs)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        model.save(model_path+model_name)
        return model

    def load(self, model_path="./model/word2vec.model"):
        model = Word2Vec.load(model_path)
        return model

    def getEmbeddedFoodList(self, x, model):
        x_ = []
        x = x.split(' ')
        for i in x:
            if '(' in i and ':' in i and ')' in i:
                continue
            if '/' in i:
                x_.extend(i.split('/'))
            else:
                x_.append(i)
        x_ = list(set(x_))
        x_.remove('')
        avg_vector = np.zeros((self.emb_dim))

        for food in x_:
            vector = model.wv.get_vector(food)
            avg_vector += vector
        avg_vector /= len(x_)
        return avg_vector
'''word2vec end'''