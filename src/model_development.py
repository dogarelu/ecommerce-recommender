from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def develop_models(user_item_matrix, meta_df):
    # Collaborative Filtering: Matrix Factorization using SVD
    svd = TruncatedSVD(n_components=100)
    matrix_factorized = svd.fit_transform(user_item_matrix)

    # Content-Based Filtering: TF-IDF on product descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(meta_df['description'])

    # Calculate cosine similarity between products
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return matrix_factorized, cosine_sim