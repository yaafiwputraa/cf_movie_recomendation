import pandas as pd
import numpy as np
import torch
import torch.optim as optim

def create_dataset(ratings: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: ratings DataFrame (userId, movieId, rating)
    Return: feedback matrix (num_movies, num_users)
    '''
    matrix = ratings.pivot(index="movieId", columns="userId", values="rating")
    matrix = matrix.fillna(0)
    return matrix

def normalizeRatings(Y: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    Y: rating matrix (num_movies, num_users)
    R: binary matrix indicating if a rating exists
    Returns Ynorm (normalized Y) and Ymean (mean ratings per movie).
    '''
    Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R)
    return Ynorm, Ymean

def prepare_data(Y_df: pd.DataFrame, R_df: pd.DataFrame, newRatings: np.ndarray = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Converts DataFrames to NumPy arrays, adds new user ratings, normalizes, and converts to PyTorch tensors.
    Y_df: DataFrame of ratings (movies x users)
    R_df: DataFrame of binary indicators (movies x users)
    newRatings: NumPy array of new user's ratings (num_movies x 1)
    Returns: Ynorm_tensor, Ymean_tensor, R_tensor
    '''
    Y_np = Y_df.to_numpy()
    R_np = R_df.to_numpy()

    if newRatings is not None:
        Y_np = np.c_[Y_np, newRatings]  # Add new ratings as a new column
        R_np = np.c_[R_np, (newRatings != 0).astype(int)] # Add new user's R values

    Ynorm, Ymean = normalizeRatings(Y_np, R_np) # Uses the numpy arrays

    Ynorm_tensor = torch.tensor(Ynorm, dtype=torch.float64)
    Ymean_tensor = torch.tensor(Ymean, dtype=torch.float64)
    R_tensor = torch.tensor(R_np, dtype=torch.float64) # Ensure R is also float for some PyTorch ops if needed, or int
    
    return Ynorm_tensor, Ymean_tensor, R_tensor

def cost_func(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, lambda_: float) -> torch.Tensor:
    '''
    Calculates the collaborative filtering cost.
    X: item features (num_movies, num_features)
    W: user parameters (num_users, num_features)
    b: user bias parameters (1, num_users)
    Y: ratings matrix (num_movies, num_users)
    R: binary matrix indicating ratings (num_movies, num_users)
    lambda_: regularization parameter
    Returns: J (cost)
    '''
    y_hat = torch.matmul(X, W.t()) + b
    j = (y_hat - Y) * R
    J = 0.5 * torch.sum(j**2) + (lambda_ / 2) * (torch.sum(W**2) + torch.sum(X**2))
    return J

def train_model_and_get_recommendations(
        Y_df: pd.DataFrame, 
        R_df: pd.DataFrame, 
        my_ratings_array: np.ndarray, 
        movies_df: pd.DataFrame, 
        num_features: int = 50, 
        num_iterations: int = 50, 
        lambda_reg: float = 1.0
    ) -> list[tuple[str, float, str]]:
    '''
    Trains the collaborative filtering model and generates recommendations for a new user.
    Y_df: DataFrame of existing ratings (movies x users)
    R_df: DataFrame of binary indicators for existing ratings
    my_ratings_array: NumPy array of the new user's ratings (num_movies length)
    movies_df: DataFrame containing all movie details (movieId, title, genres)
    num_features: Number of latent features
    num_iterations: Number of training iterations
    lambda_reg: Regularization parameter
    Returns: List of tuples: (movie_title, predicted_rating, movie_genres)
    '''
    num_movies = Y_df.shape[0]
    # Add 1 to existing users for the new user being added
    num_users = Y_df.shape[1] + 1 
    
    # Initialize model parameters (features for movies, weights for users, bias for users)
    X = torch.randn((num_movies, num_features), dtype=torch.float64, requires_grad=True)
    W = torch.randn((num_users, num_features), dtype=torch.float64, requires_grad=True)
    b = torch.randn(1, num_users, dtype=torch.float64, requires_grad=True) # Bias for each user

    # Prepare data: Convert to tensors, normalize, and add new user ratings.
    # my_ratings_array should be a column vector for prepare_data
    my_ratings_col_vec = my_ratings_array.reshape(-1, 1)
    Ynorm, Ymean, R_tensor = prepare_data(Y_df, R_df, my_ratings_col_vec)

    # Optimizer
    opt = optim.Adam([X, W, b], lr=1e-1)

    # Training loop
    for iter_loop in range(num_iterations):
        opt.zero_grad()
        cost = cost_func(X, W, b, Ynorm, R_tensor, lambda_reg)
        cost.backward()
        opt.step()
        # if iter_loop % 10 == 0:
        #     print(f"Iteration {iter_loop+1}/{num_iterations}, Cost: {cost.item():.2f}")

    # Generate predictions
    pred = torch.matmul(X, W.t()) + b  # Predictions for all users
    predm = pred + Ymean  # Add back the mean to get actual rating scale

    # Get predictions for the new user (the last column in the augmented matrix)
    my_pred = predm[:, -1]
    
    # Sort predictions in descending order
    idx_pred = torch.argsort(my_pred, descending=True)

    recommendations_list = []
    count_disp = 0
    for i_val_tensor in idx_pred:
        i_val = i_val_tensor.item() # Convert tensor index to integer
        # Check if the movie at this index has been rated by the new user
        # my_ratings_array corresponds to the original movie indices if Y_df.index is consistent.
        if my_ratings_array[i_val] == 0 and count_disp < 10:
            # Get movieID from the original Y_df's index
            movie_id_val = Y_df.index[i_val] 
            
            movie_info = movies_df[movies_df["movieId"] == movie_id_val]
            if not movie_info.empty:
                title_rec = movie_info["title"].values[0]
                genres_rec = movie_info["genres"].values[0] if "genres" in movie_info.columns else "Unknown"
                predicted_rating_rec = my_pred[i_val].item() # Get Python number
                
                recommendations_list.append((title_rec, predicted_rating_rec, genres_rec))
                count_disp += 1
        
    return recommendations_list
