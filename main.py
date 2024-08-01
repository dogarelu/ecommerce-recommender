import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data
from src.model_development import develop_models
from src.recommendation_engine import get_recommendations
from src.evaluation import evaluate_model, ab_test

def main():
    # Load and preprocess data
    df, user_item_matrix, meta_df = preprocess_data()

    # Develop models
    matrix_factorized, cosine_sim = develop_models(user_item_matrix, meta_df)

    # Evaluate the model
    rmse = evaluate_model(df)
    print(f"RMSE: {rmse:.4f}")

    # Get recommendations for a sample user
    user_id = df['user_id'].iloc[0]
    recommendations = get_recommendations(user_id, user_item_matrix, matrix_factorized, cosine_sim, meta_df)
    print(f"Top 10 recommendations for user {user_id}:")
    for product_id, score in recommendations:
        product_title = meta_df[meta_df['product_id'] == product_id]['title'].values[0]
        print(f"{product_title}: {score:.2f}")

    # Perform A/B testing
    ab_test_recommendations = ab_test(user_id, user_item_matrix, meta_df)
    print(f"A/B test recommendations for user {user_id}:")
    for product_id in ab_test_recommendations:
        product_title = meta_df[meta_df['product_id'] == product_id]['title'].values[0]
        print(product_title)

if __name__ == "__main__":
    main()