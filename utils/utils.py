import pandas as pd
import json 
from fastapi import HTTPException
import pymongo

def delete_all_predictions():
    client = pymongo.MongoClient("mongodb+srv://qh2023:1@clusterbegin0.iojcaoc.mongodb.net") 
    db = client["film_application"]
    collection = db["predictions"]
    collection.delete_many({})

def upload_to_mongo(data):
    '''
    data = [
        {'user_id': [1, 1, 2, 2], 'movie_id': [101, 102, 101, 103], 'predictions': [4.5, 3.0, 2.5, 4.0]},
        {'user_id': [2, 2, 3, 3], 'movie_id': [104, 102, 101, 103], 'predictions': [4.5, 3.0, 2.5, 4.0]},
    ]
    '''
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://qh2023:1@clusterbegin0.iojcaoc.mongodb.net") 
    db = client["film_application"]
    collection = db["predictions"]

    # Iterate over the data and insert documents into the collection
    for entry in data:
        # Iterate over each user, movie, and prediction simultaneously
        for user_id, movie_id, prediction in zip(entry['user_id'], entry['movie_id'], entry['predictions']):
            # Query to check if the user document already exists
            user_query = {'user_id': user_id}
            user_document = collection.find_one(user_query)

            # If the user document exists, update it with the new movie and prediction
            if user_document:
                collection.update_one(
                    {"user_id": user_id},
                    {"$push": {
                        "movies.$[].movie_id": movie_id,
                        "movies.$[].prediction": prediction
                    }}
                )
            # If the user document doesn't exist, create a new document
            else:
                user_entry = {
                    'user_id': user_id,
                    'movies': [{'movie_id': [movie_id], 'prediction': [prediction]}]
                }
                collection.insert_one(user_entry)

    # Close the MongoDB connection
    client.close()
    
def read_data(table_name, table_columns=None):
    return pd.read_csv(
        f"data\dataset\{table_name}.dat",
        delimiter="::",
        engine="python",
        header=None,
        names=table_columns,
        encoding="latin-1",
    )
    
def check_user_id_availability(user_id: int):
    client = pymongo.MongoClient("mongodb+srv://qh2023:1@clusterbegin0.iojcaoc.mongodb.net") 
    db = client["film_application"]
    collection = db["predictions"]
    user_query = {'user_id': user_id}
    user_document = collection.find_one(user_query)

    if not user_document:
        raise HTTPException(status_code=404, detail="User ID not found")
    return user_id