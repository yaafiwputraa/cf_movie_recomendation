# Collaborative Filtering Movie Recommendation

A movie recommendation system using collaborative filtering with matrix factorization and an interactive Streamlit interface.

## Project Overview

This project implements a **collaborative filtering** system to recommend movies using **matrix factorization**, trained with PyTorch. Users can provide ratings through a **Streamlit** interface, and the system returns personalized recommendations based on learned user-item interactions.

## Features

* Matrix Factorization using PyTorch for rating prediction
* Rating normalization & L2 regularization for better generalization
* Training optimization using Adam optimizer
* Interactive movie rating interface via **Streamlit**
* Real-time movie recommendation for new users

## How to Use

1. Run the app: `streamlit run app.py`
2. Rate a few movies using the interface
3. Get movie recommendations based on your preferences

## Sample Backend Code

### Cost Function

```python
def cost_func(X, W, b, Y, R, lambda_):
    y_hat = torch.matmul(X,W.t()) + b 
    j = (y_hat - Y) * R
    J = 0.5 * torch.sum(j**2) + lambda_/2 * (torch.sum(W**2) + torch.sum(X**2)) 
    return J
```

### Normalize Ratings

```python
def normalizeRatings(Y, R):
    Ymean = (np.sum(Y*R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return Ynorm, Ymean
```

## Model Parameters

| Parameter       | Description                      | Default |
| --------------- | -------------------------------- | ------- |
| `num_features`  | Number of latent features        | 150     |
| `lambda_`       | L2 regularization coefficient    | 1       |
| `iterations`    | Number of training iterations    | 200     |
| `learning_rate` | Learning rate for Adam optimizer | 0.1     |

## Tech Stack

* Python
* PyTorch
* NumPy, Pandas
* Streamlit

## Output

* Predicts unseen movie ratings for a user
* Returns top-N movie recommendations
* Interactive UI for rating input and personalized recommendations
