import pytest
import pandas as pd
import numpy as np
import torch
from src.recommendation_logic import (
    create_dataset,
    normalizeRatings,
    prepare_data,
    cost_func,
    train_model_and_get_recommendations
)

# Fixtures for common data structures
@pytest.fixture
def sample_ratings_df():
    # movieId, userId, rating
    data = {
        'userId': [1, 1, 2, 2, 3, 3, 3, 4, 4, 1],
        'movieId': [1, 2, 1, 3, 2, 3, 4, 4, 1, 4], # User 1 rated movie 4
        'rating': [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 3.0, 4.0, 1.0, 5.0] 
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_movies_df():
    data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': ['Action|Adventure', 'Comedy', 'Drama', 'Action|Thriller', 'Sci-Fi']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_y_df(sample_ratings_df):
    # This creates the pivoted Y_df (movies x users)
    # Movie IDs will be index: 1, 2, 3, 4
    # User IDs will be columns: 1, 2, 3, 4
    # Expected Y_df from sample_ratings_df:
    # movieId  1    2    3    4
    # 1        5.0  4.0  0.0  1.0
    # 2        3.0  0.0  5.0  0.0
    # 3        0.0  2.0  4.0  0.0
    # 4        5.0  0.0  3.0  4.0 
    return create_dataset(sample_ratings_df)

@pytest.fixture
def sample_r_df(sample_y_df):
    return (sample_y_df != 0).astype(int)

# Tests for create_dataset
def test_create_dataset(sample_ratings_df):
    y_df = create_dataset(sample_ratings_df)
    assert isinstance(y_df, pd.DataFrame)
    # Movie IDs: 1, 2, 3, 4. User IDs: 1, 2, 3, 4
    assert y_df.shape == (4, 4) 
    assert y_df.loc[1, 1] == 5.0  # Movie 1, User 1
    assert y_df.loc[1, 2] == 4.0  # Movie 1, User 2
    assert y_df.loc[1, 3] == 0.0  # Movie 1, User 3 (no rating)
    assert y_df.loc[2, 3] == 5.0  # Movie 2, User 3
    assert y_df.loc[4, 1] == 5.0  # Movie 4, User 1

# Tests for normalizeRatings
def test_normalize_ratings():
    Y = np.array([[5, 0, 1], [3, 5, 0], [0, 4, 4]], dtype=float)
    R = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int)
    
    Ynorm_expected_mean_m1 = (5+1)/2.0 # Movie 1 mean = 3
    Ynorm_expected_mean_m2 = (3+5)/2.0 # Movie 2 mean = 4
    Ynorm_expected_mean_m3 = (4+4)/2.0 # Movie 3 mean = 4

    Ynorm_expected = np.array([
        [5-Ynorm_expected_mean_m1, 0, 1-Ynorm_expected_mean_m1],
        [3-Ynorm_expected_mean_m2, 5-Ynorm_expected_mean_m2, 0],
        [0, 4-Ynorm_expected_mean_m3, 4-Ynorm_expected_mean_m3]
    ])
    
    Ynorm, Ymean = normalizeRatings(Y, R)
    
    assert Ymean.shape == (3, 1)
    assert np.allclose(Ymean, np.array([[Ynorm_expected_mean_m1], [Ynorm_expected_mean_m2], [Ynorm_expected_mean_m3]]))
    assert np.allclose(Ynorm, Ynorm_expected)
    # Check that means of rated items in Ynorm are close to zero
    for i in range(Ynorm.shape[0]):
        rated_norm_values = Ynorm[i, R[i, :] == 1]
        if len(rated_norm_values) > 0:
            assert np.isclose(np.mean(rated_norm_values), 0.0)

# Tests for prepare_data
def test_prepare_data(sample_y_df, sample_r_df):
    # Test without new ratings
    ynorm_tensor, ymean_tensor, r_tensor = prepare_data(sample_y_df, sample_r_df)
    
    assert isinstance(ynorm_tensor, torch.Tensor)
    assert isinstance(ymean_tensor, torch.Tensor) # Corrected typo
    assert isinstance(r_tensor, torch.Tensor)
    
    assert ynorm_tensor.shape == sample_y_df.shape
    assert ymean_tensor.shape == (sample_y_df.shape[0], 1)
    assert r_tensor.shape == sample_r_df.shape

    # Test with new ratings
    # sample_y_df has 4 movies. New user rates 2 of them.
    new_user_ratings = np.array([5.0, 0.0, 3.0, 0.0]).reshape(-1,1) 
    
    ynorm_tensor_new, ymean_tensor_new, r_tensor_new = prepare_data(sample_y_df, sample_r_df, new_user_ratings)
    
    assert ynorm_tensor_new.shape == (sample_y_df.shape[0], sample_y_df.shape[1] + 1)
    assert ymean_tensor_new.shape == (sample_y_df.shape[0], 1) # Ymean is per movie, shape doesn't change by adding users
    assert r_tensor_new.shape == (sample_r_df.shape[0], sample_r_df.shape[1] + 1)
    
    # Check if the new user's R values are correct in the last column of r_tensor_new
    expected_r_new_user = (new_user_ratings.flatten() != 0).astype(float) # As R_tensor is float
    assert torch.allclose(r_tensor_new[:, -1], torch.tensor(expected_r_new_user, dtype=torch.float64))


# Tests for cost_func
def test_cost_func():
    # num_movies=2, num_users=1, num_features=1
    X = torch.tensor([[0.5], [1.0]], dtype=torch.float64, requires_grad=True)
    W = torch.tensor([[0.8]], dtype=torch.float64, requires_grad=True) # 1 user, 1 feature
    b = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True) # bias for 1 user
    
    # User 1 rated movie 1 as 4, movie 2 as 2
    Y = torch.tensor([[4.0], [2.0]], dtype=torch.float64) 
    R = torch.tensor([[1.0], [1.0]], dtype=torch.float64) # Both movies rated
    lambda_ = 0.1

    # Predictions:
    # Movie 1: (0.5 * 0.8) + 0.1 = 0.4 + 0.1 = 0.5
    # Movie 2: (1.0 * 0.8) + 0.1 = 0.8 + 0.1 = 0.9
    
    # Errors (y_hat - Y):
    # Movie 1: 0.5 - 4.0 = -3.5
    # Movie 2: 0.9 - 2.0 = -1.1
    
    # Squared errors * R:
    # Movie 1: (-3.5)**2 * 1 = 12.25
    # Movie 2: (-1.1)**2 * 1 = 1.21
    
    # Sum of squared errors = 12.25 + 1.21 = 13.46
    # J_unreg = 0.5 * 13.46 = 6.73
    
    # Regularization:
    # Reg_W = sum(W^2) = 0.8**2 = 0.64
    # Reg_X = sum(X^2) = 0.5**2 + 1.0**2 = 0.25 + 1.0 = 1.25
    # J_reg = (lambda_ / 2) * (Reg_W + Reg_X) = (0.1 / 2) * (0.64 + 1.25) = 0.05 * 1.89 = 0.0945
    
    # Total cost J = J_unreg + J_reg = 6.73 + 0.0945 = 6.8245
    expected_cost = 6.8245
    
    cost = cost_func(X, W, b, Y, R, lambda_)
    assert isinstance(cost, torch.Tensor)
    assert torch.isclose(cost, torch.tensor(expected_cost, dtype=torch.float64))

# Tests for train_model_and_get_recommendations
def test_train_model_and_get_recommendations(sample_y_df, sample_r_df, sample_movies_df):
    # sample_y_df has 4 movies, 4 users. Movie IDs are 1,2,3,4
    # sample_movies_df has 5 movies. Movie IDs are 1,2,3,4,5
    
    # New user ratings for the 4 movies in sample_y_df
    # Corresponds to movies with IDs 1, 2, 3, 4 based on sample_y_df.index
    my_ratings = np.array([5.0, 0.0, 0.0, 1.0]) # Rated movie 1 and movie 4
                                                # Movies 2 and 3 are unrated by new user

    # To ensure movies_df aligns with Y_df's movie indices for the test
    # We only need movies that are in Y_df for the my_ratings_array mapping.
    # The function itself will use the full movies_df to lookup titles/genres.
    
    torch.manual_seed(42) # For reproducibility of randn
    np.random.seed(42)

    recommendations = train_model_and_get_recommendations(
        sample_y_df,       # Y_df from fixture (4 movies, 4 users)
        sample_r_df,       # R_df from fixture
        my_ratings,        # New user's ratings for these 4 movies
        sample_movies_df,  # Full movies table for title/genre lookup
        num_features=3,    # Small number of features for faster test
        num_iterations=5   # Small number of iterations for faster test
    )
    
    assert isinstance(recommendations, list)
    if recommendations: # If list is not empty
        for rec in recommendations:
            assert isinstance(rec, tuple)
            assert len(rec) == 3
            assert isinstance(rec[0], str)  # title
            assert isinstance(rec[1], float) # predicted_rating
            assert isinstance(rec[2], str)  # genres

            # Check that recommended movies are not among those already rated by the new user
            # Movie titles for IDs 2 and 3 (which were unrated) are 'Movie B', 'Movie C'
            # Movie titles for IDs 1 and 4 (which were rated) are 'Movie A', 'Movie D'
            assert rec[0] != 'Movie A' 
            assert rec[0] != 'Movie D'

    # Check that it returns up to 10 recommendations
    assert len(recommendations) <= 10
    
    # Example: Test with a Y_df that has no movies (should probably return empty or handle gracefully)
    empty_y_df = pd.DataFrame(columns=sample_y_df.columns, index=pd.Index([], name='movieId'))
    empty_r_df = pd.DataFrame(columns=sample_y_df.columns, index=pd.Index([], name='movieId'))
    empty_my_ratings = np.array([]) # Empty ratings for no movies
    
    recommendations_empty = train_model_and_get_recommendations(
        empty_y_df, empty_r_df, empty_my_ratings, sample_movies_df, 
        num_features=2, num_iterations=2
    )
    assert recommendations_empty == []

    # Example: Test when all movies are rated by the new user (should return empty list)
    all_rated_my_ratings = np.array([5.0, 4.0, 3.0, 2.0])
    recommendations_all_rated = train_model_and_get_recommendations(
        sample_y_df, sample_r_df, all_rated_my_ratings, sample_movies_df,
        num_features=2, num_iterations=2
    )
    assert recommendations_all_rated == []
