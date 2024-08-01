import pandas as pd
import numpy as np

def preprocess_data():
    # Load the dataset
    reviews_df = pd.read_json('data/reviews_Electronics_5.json', lines=True)
    meta_df = pd.read_json('data/meta_Electronics.json', lines=True)

    # Preprocess the data
    reviews_df = reviews_df[['reviewerID', 'asin', 'overall']]
    reviews_df.columns = ['user_id', 'product_id', 'rating']

    meta_df = meta_df[['asin', 'title', 'category', 'description']]
    meta_df.columns = ['product_id', 'title', 'category', 'description']

    # Merge reviews and metadata
    df = pd.merge(reviews_df, meta_df, on='product_id')

    # Handle missing data
    df = df.dropna()

    # Create a user-item interaction matrix
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)

    return df, user_item_matrix, meta_df