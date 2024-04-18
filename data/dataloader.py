import os
import re
import sys

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wikipedia
from torch.utils.data import Dataset, DataLoader
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
        else:
            return len(self.val_ratings)
  
    def __getitem__(self, idx):
        if self.train_val == 'train':
            data = self.train_ratings.iloc[idx]
        if self.train_val == 'val':
            data = self.val_ratings.iloc[idx]
        user_id, movie_id, label = data
        user_data = self.users[self.users.user_id==user_id].values.squeeze()
        movie_data = self.movies[self.movies.movie_id==movie_id].values.squeeze()
        return torch.FloatTensor(user_data), torch.FloatTensor(movie_data), torch.FloatTensor([label]), torch.LongTensor([user_id])
    
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
