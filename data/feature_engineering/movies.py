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


class ProcessMovies:
    def __init__(self, output_folder):
        self.output_folder = output_folder
    
    def __get_wikipedia_page_name(self, raw_name):
        names = wikipedia.search(raw_name)
        if len(names) == 0:
            return ''
        else:
            return names[0]

    def __get_movie_plot(self, page_name):            
        try:
            movie_page_content = str(wikipedia.page(page_name, auto_suggest=False).content)
            for keyword in ['Plot', 'Synopsis']:
                re_groups = re.search(f"{keyword} ==(.*?)=+ [A-Z]", str(movie_page_content).replace('\n', ''))
                if re_groups:
                    return re_groups.group(1)
                paragraphs = re.split(r'\n+', movie_page_content.strip())
                for paragraph in paragraphs:
                    if paragraph.strip():
                        return paragraph.strip()
        except:
            return ''

    def __bert_emb(self, text, tokenizer, model, device):
        if len(text) <= 2 or text is None:  
            return np.zeros((1, 768))

        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens]).to(device)  # Move tensor to GPU
        with torch.no_grad():
            outputs = model(tokens_tensor)
            embeddings = outputs[0][:, 1:-1, :]  # Remove [CLS] and [SEP] tokens
            embedding = torch.mean(embeddings, dim=1)
        return embedding.cpu().numpy()  # Move embedding back to CPU

    def __umap_dim_reduction(self, embeddings, n_components=32):
        reducer = umap.UMAP(n_components=n_components)
        embedding_umap = reducer.fit_transform(embeddings)
        return embedding_umap
    
    def __get_all_movie_plots(self, movies):
        title_list = movies['title']
        wikipedia_page_names = [self.__get_wikipedia_page_name(name) for name in tqdm(title_list)]
        movie_plots = [self.__get_movie_plot(page_name) for page_name in tqdm(wikipedia_page_names)]
        return movie_plots
    
    def __get_plot_embedding(self, movies):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        try:
            movie_plots = movies['movie_plot']
        except:
            movie_plots = self.__get_all_movie_plots(movies)
            movies['movie_plot'] = movie_plots
        plot_embs = [self.__bert_emb(i[:512], tokenizer, model, device) for i in tqdm(movie_plots)]
        plot_embs = np.concatenate(plot_embs, axis=0)
        plot_embs_reduced = self.__umap_dim_reduction(plot_embs)
        plot_embs_reduced = pd.DataFrame(plot_embs_reduced, columns=[f'emb_{i}' for i in range(plot_embs_reduced.shape[1])])
        prev_len = len(movies)
        movies = pd.concat([movies.reset_index(drop=True), plot_embs_reduced], axis=1)
        assert len(movies) == prev_len
        return movies
    
    def __process_genre(self, movies):
        movies['genre'] = movies['genre'].apply(lambda x: x.split('|'))
        genre_dummies = pd.get_dummies(movies['genre'].explode()).groupby(level=0).sum()
        movies = pd.concat([movies, genre_dummies], axis=1)
        return movies

    def main(self):
        movies = read_data('movies', ['movie_id', 'title', 'genre'])
        movies['year'] = movies['title'].apply(lambda title: re.search('\((\d*)\)', title).groups(1)[0])
        movies['year'] = movies['year'].astype(int)
        k=10
        for i in range(k):
            movies_tmp = movies.iloc[len(movies)//k*i: len(movies)//k*(i+1)]
            movies_tmp = self.__process_genre(movies_tmp)
            movies_tmp = self.__get_plot_embedding(movies_tmp)
            movies_tmp = movies_tmp.drop(['title', 'genre','wikipedia_page_name','movie_plot'], axis=1, errors='ignore')
            movies_tmp.to_parquet(os.path.join(self.output_folder, f'movies_{i}.pqt'))
        dfs = [pd.read_parquet(os.path.join(self.output_folder, f'movies_{i}.pqt')) for i in range(k)]
        dfs = pd.concat(dfs, ignore_index=True)
        dfs.to_parquet(os.path.join(self.output_folder, f'movies.pqt'))

if __name__=='__main__':
    ProcessMovies(r'data\dataset').main()
