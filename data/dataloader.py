import os
import re
import sys

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wikipedia
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
from data import ProcessMovies, ProcessRatings
from utils import read_data

np.random.seed(123)

class MovieLensDataset(Dataset):
    def __init__(self, train_val='train'):
        '''
        part: train/val
        '''
        self.movies, self.users, self.train_ratings, self.val_ratings = self.__feature_engineering()
        self.train_val=train_val
        
    def __len__(self):
        if self.train_val == 'train':
            return len(self.train_ratings)
        if self.train_val == 'val':
            return len(self.val_ratings)
  
    def __getitem__(self, idx):
        if self.train_val == 'train':
            data = self.train_ratings.iloc[idx]
        if self.train_val == 'val':
            data = self.val_ratings.iloc[idx]
        user_id, movie_id, label = data
        user_data = self.users[self.users.user_id==user_id].values.squeeze()
        movie_data = self.movies[self.movies.movie_id==movie_id].values.squeeze()
        return torch.LongTensor(user_data), torch.FloatTensor(movie_data), torch.LongTensor([user_id]), torch.FloatTensor([label])
    
    def __feature_engineering(self):
        if not os.path.exists(r'data\dataset\movies.pqt'):
            ProcessMovies(r'data\dataset').main()
        movies = pd.read_parquet(r'data\dataset\movies.pqt')
        train_ratings, val_ratings, ratings_ft = ProcessRatings(r'data\dataset').main()
        movies = movies.merge(ratings_ft)
        users = read_data('users',table_columns=['user_id','gender','age', 'occupation', 'zipcode'])
        users = users.drop('zipcode',axis=1)
        users['gender'] = [1.0 if i=='M' else 0.0 for i in users['gender']]
        return movies, users, train_ratings, val_ratings

class MovieLensSingleUserDataset(Dataset):
    def __init__(self, user_id, train_infer='train'):
        '''
        part: train/infer
        '''
        self.train_infer = train_infer
        self.user_id = user_id
        self.movies, self.users, self.ratings = self.__feature_engineering()
        
    def __len__(self):
        if self.train_infer == 'train':
            return len(self.ratings)
        else:
            return len(self.movies)
  
    def __getitem__(self, idx):
        if self.train_infer == 'train':
            data = self.ratings.iloc[idx]
            user_id, movie_id, label = data
            user_data = self.users[self.users.user_id==user_id].values.squeeze()
            movie_data = self.movies[self.movies.movie_id==movie_id].values.squeeze()
            return torch.LongTensor(user_data), torch.FloatTensor(movie_data), torch.LongTensor([user_id]), torch.FloatTensor([label])
        if self.train_infer == 'infer':
            movie_data = self.movies.iloc[idx].values.squeeze()
            user_data = self.users[self.users.user_id==self.user_id].values.squeeze()
            return torch.LongTensor(user_data), torch.FloatTensor(movie_data), torch.LongTensor([self.user_id])

    def __feature_engineering(self):
        if not os.path.exists(r'data\dataset\movies.pqt'):
            ProcessMovies(r'data\dataset').main()
        movies = pd.read_parquet(r'data\dataset\movies.pqt')
        ratings, ratings_ft = ProcessRatings(r'data\dataset').get_rating_per_user(self.user_id)
        movies = movies.merge(ratings_ft)
        users = read_data('users',table_columns=['user_id','gender','age', 'occupation', 'zipcode'])
        users = users[users.user_id==self.user_id].reset_index(drop=True)
        users = users.drop('zipcode',axis=1)
        users['gender'] = [1.0 if i=='M' else 0.0 for i in users['gender']]
        return movies, users, ratings

class MovieLensInferAllDataset(Dataset):
    def __init__(self):
        '''
        part: train/infer
        '''
        self.movies, self.users = self.__feature_engineering()
        
    def __len__(self):
        return len(self.movies) * len(self.users)
  
    def __getitem__(self, idx):
        movie_idx = idx % len(self.movies)
        user_idx = idx // len(self.users)
        movie_id = self.movies.iloc[movie_idx]['movie_id']
        user_id = self.users.iloc[user_idx]['user_id']
        movie_data = self.movies.iloc[movie_idx].values.squeeze()
        user_data = self.users.iloc[user_idx].values.squeeze()
        return torch.LongTensor(user_data), torch.FloatTensor(movie_data), torch.LongTensor([user_id]), torch.LongTensor([movie_id])

    def __feature_engineering(self):
        if not os.path.exists(r'data\dataset\movies.pqt'):
            ProcessMovies(r'data\dataset').main()
        movies = pd.read_parquet(r'data\dataset\movies.pqt')
        ratings_ft = ProcessRatings(r'data\dataset').get_ratings_ft()
        movies = movies.merge(ratings_ft)
        users = read_data('users',table_columns=['user_id','gender','age', 'occupation', 'zipcode'])
        users = users.drop('zipcode',axis=1)
        users['gender'] = [1.0 if i=='M' else 0.0 for i in users['gender']]
        return movies, users
