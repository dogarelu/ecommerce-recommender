# E-commerce Recommendation Engine

This project implements an AI-powered recommendation engine for an e-commerce platform. It provides personalized product recommendations to users based on their browsing and purchase history, as well as other relevant data.

## Features

- Hybrid recommendation system combining collaborative and content-based filtering
- Handles new users with popularity-based recommendations
- Evaluation mechanism with cross-validation and multiple metrics
- A/B testing capability for comparing recommendation strategies

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

## Installation

1. Clone the repository:
git clone https://github.com/dogarelu/ecommerce-recommender.git

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

## Data Preparation

1. Run the data download script:
python download_data.py
This will download and extract the Amazon Electronics dataset to the `data/` directory.

## Usage

1. Run the main script to preprocess data, train models, and generate recommendations:
python main.py
This script will:

- Preprocess the data
- Develop the recommendation models
- Evaluate the model's performance
- Generate sample recommendations for a user
- Perform an A/B test

2. To use the recommendation engine in your own code:

```python
from src.data_preprocessing import preprocess_data
from src.model_development import develop_models
from src.recommendation_engine import get_recommendations

# Preprocess data
df, user_item_matrix, meta_df = preprocess_data()

# Develop models
matrix_factorized, cosine_sim = develop_models(user_item_matrix, meta_df)

# Get recommendations for a user
user_id = 'A2SUAM1J3GNN3B'  # Replace with actual user ID
recommendations = get_recommendations(user_id, user_item_matrix, matrix_factorized, cosine_sim, meta_df)

# Print recommendations
for product_id, score in recommendations:
    product_title = meta_df[meta_df['product_id'] == product_id]['title'].values[0]
    print(f"{product_title}: {score:.2f}")
```

## Evaluation
The system uses the following metrics for evaluation:

RMSE (Root Mean Square Error)
MAE (Mean Absolute Error)
NDCG (Normalized Discounted Cumulative Gain)

To run the evaluation separately:
```python
from src.evaluation import evaluate_model
from src.recommendation_engine import get_recommendations

evaluation_results = evaluate_model(df, user_item_matrix, get_recommendations)
print(evaluation_results)
```

## A/B Testing
The system includes a simple A/B testing mechanism. To use it:

```python
from src.evaluation import ab_test

user_id = 'A2SUAM1J3GNN3B'  # Replace with actual user ID
ab_test_recommendations = ab_test(user_id, user_item_matrix, meta_df)
```

## Customization

To modify the recommendation algorithm, edit the get_recommendations function in src/recommendation_engine.py.
To change the evaluation metrics or process, modify the evaluate_model function in src/evaluation.py.
To alter the data preprocessing steps, update the preprocess_data function in src/data_preprocessing.py.