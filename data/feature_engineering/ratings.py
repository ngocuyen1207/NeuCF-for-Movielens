import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import umap
import wikipedia
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

sys.path.append(os.getcwd())
from utils import read_data

class ProcessRatings:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def __fe_rating_features(self, ratings):
        ratings_ft = pd.DataFrame(ratings.groupby('movie_id')['rating'].mean())
        ratings_ft.columns=['rating_mean']
        ratings_ft['rating_counts'] = ratings.groupby('movie_id')['rating'].count()
        return ratings_ft.reset_index()
    
    def __get_label(self, ratings):
        ratings['label'] = [1 if i >= 4 else 0 for i in ratings.rating]
        print('Ratio of class 1: ', sum(ratings['label'])/len(ratings))
        ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'] \
                                        .rank(method='first', ascending=False)
        test_item = 5
        train_ratings = ratings[ratings['rank_latest'] > test_item]
        test_ratings = ratings[ratings['rank_latest'] <= test_item]

        train_ratings = train_ratings[['user_id','movie_id','label']]
        test_ratings = test_ratings[['user_id','movie_id','label']]
        return train_ratings, test_ratings

    def __remove_ratings_not_in_movie_list(self, ratings):
        movie_list = pd.read_parquet(r'C:\Users\uyen\OneDrive\NeuralCF\data\dataset\movies.pqt', columns=['movie_id']).values.squeeze()
        ratings = ratings[ratings.movie_id.isin(movie_list)].reset_index(drop=True)
        return ratings

    def main(self):
        ratings = read_data('ratings', ['user_id', 'movie_id', 'rating', 'timestamp'])
        ratings = self.__remove_ratings_not_in_movie_list(ratings)
        ratings_ft = self.__fe_rating_features(ratings)
        train_ratings, test_ratings = self.__get_label(ratings)
        return train_ratings, test_ratings, ratings_ft
    
    def get_rating_per_user(self, user_id):
        ratings = read_data('ratings', ['user_id', 'movie_id', 'rating', 'timestamp'])
        ratings_ft = self.__fe_rating_features(ratings)

        ratings = ratings[ratings.user_id == user_id].reset_index(drop=True)
        ratings = self.__remove_ratings_not_in_movie_list(ratings)
        ratings['label'] = [1 if i >= 4 else 0 for i in ratings.rating]
        ratings = ratings[['user_id','movie_id','label']]
        return ratings, ratings_ft
