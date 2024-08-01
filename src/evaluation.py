from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
import numpy as np
import random

def evaluate_model(df, user_item_matrix, get_recommendations_func, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    ndcg_scores = []
    
    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        
        # Create user-item matrix for training data
        train_matrix = train_data.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
        
        y_true = []
        y_pred = []
        ndcg_true = []
        ndcg_pred = []
        
        for user_id in test_data['user_id'].unique():
            user_test_data = test_data[test_data['user_id'] == user_id]
            
            if user_id in train_matrix.index:
                recommendations = get_recommendations_func(user_id, train_matrix, None, None, None)
                rec_dict = dict(recommendations)
                
                for _, row in user_test_data.iterrows():
                    true_rating = row['rating']
                    pred_rating = rec_dict.get(row['product_id'], 0)
                    
                    y_true.append(true_rating)
                    y_pred.append(pred_rating)
                
                ndcg_true.append(user_test_data['rating'].values)
                ndcg_pred.append([rec_dict.get(pid, 0) for pid in user_test_data['product_id']])
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        ndcg = ndcg_score(ndcg_true, ndcg_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        ndcg_scores.append(ndcg)
    
    return {
        'RMSE': np.mean(rmse_scores),
        'MAE': np.mean(mae_scores),
        'NDCG': np.mean(ndcg_scores)
    }

def ab_test(user_id, user_item_matrix, meta_df, test_percentage=10):
    if random.randint(1, 100) <= test_percentage:
        return get_alternative_recommendations(user_id, user_item_matrix, meta_df)
    else:
        return get_recommendations(user_id, user_item_matrix, None, None, meta_df)

def get_alternative_recommendations(user_id, user_item_matrix, meta_df, n=10):
    # Implement an alternative recommendation algorithm here
    # For simplicity, we'll just return random recommendations
    return random.sample(list(user_item_matrix.columns), n)