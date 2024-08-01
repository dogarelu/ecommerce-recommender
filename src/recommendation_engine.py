import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(user_id, user_item_matrix, matrix_factorized, cosine_sim, meta_df, n=10):
    if user_id not in user_item_matrix.index or user_item_matrix.loc[user_id].sum() == 0:
        return get_popular_items(meta_df, n)
    
    # Collaborative filtering recommendations
    user_vector = matrix_factorized[user_item_matrix.index.get_loc(user_id)]
    similarities = cosine_similarity([user_vector], matrix_factorized)[0]
    similar_users = similarities.argsort()[::-1][1:11]  # Top 10 similar users
    
    cf_recommendations = set()
    for similar_user in similar_users:
        user_products = user_item_matrix.iloc[similar_user].nlargest(5).index
        cf_recommendations.update(user_products)
    
    # Content-based filtering recommendations
    user_history = user_item_matrix.loc[user_id]
    user_products = user_history[user_history > 0].index
    cb_recommendations = set()
    for product in user_products:
        product_index = meta_df[meta_df['product_id'] == product].index[0]
        similar_products = cosine_sim[product_index].argsort()[::-1][1:6]  # Top 5 similar products
        cb_recommendations.update(meta_df.iloc[similar_products]['product_id'])
    
    # Combine and rank recommendations
    all_recommendations = list(cf_recommendations.union(cb_recommendations))
    recommendation_scores = []
    for product in all_recommendations:
        cf_score = similarities[user_item_matrix.columns.get_loc(product)]
        cb_score = max([cosine_sim[meta_df[meta_df['product_id'] == p].index[0]][meta_df[meta_df['product_id'] == product].index[0]] for p in user_products])
        recommendation_scores.append((product, 0.7 * cf_score + 0.3 * cb_score))
    
    recommendation_scores.sort(key=lambda x: x[1], reverse=True)
    return recommendation_scores[:n]

def get_popular_items(meta_df, n=10):
    # Simple popularity-based recommendation for new users
    popular_items = meta_df.groupby('product_id')['rating'].mean().sort_values(ascending=False)
    return [(pid, score) for pid, score in popular_items.head(n).items()]