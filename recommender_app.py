import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Custom CSS for clean and elegant styling
st.markdown("""
<style>
    .movie-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: box-shadow 0.3s ease;
    }
    
    .movie-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .movie-title {
        font-size: 18px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
    }
    
    .movie-genre {
        font-size: 14px;
        color: #666;
        margin-bottom: 15px;
    }
    
    .rating-buttons {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    .rating-btn {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #495057;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .rating-btn:hover {
        background: #e9ecef;
        border-color: #adb5bd;
    }
    
    .rating-btn.selected {
        background: #007bff;
        border-color: #007bff;
        color: white;
    }
    
    .main-header {
        text-align: center;
        color: #333;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    .recommendation-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 4px solid #007bff;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .rec-title {
        font-size: 18px;
        font-weight: 600;
        color: #333 !important;
        margin-bottom: 8px;
    }
    
    .rec-rating {
        color: #007bff !important;
        font-weight: 500;
        margin-bottom: 8px;
        font-size: 15px;
    }
    
    .rec-genre {
        color: #666 !important;
        font-size: 14px;
    }
    
    .current-rating {
        background: #e3f2fd;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 14px;
        color: #1976d2;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

def create_dataset(ratings):
    matrix = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
    return matrix

def normalizeRatings(Y, R):
    Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R)
    return Ynorm, Ymean

def prepare_data(Y_df, R_df, newRatings=None):
    Y_np = Y_df.to_numpy()
    R_np = R_df.to_numpy()
    if newRatings is not None:
        Y_np = np.c_[Y_np, newRatings]
        R_np = np.c_[R_np, (newRatings != 0).astype(int)]
    Ynorm, Ymean = normalizeRatings(Y_np, R_np)
    return torch.tensor(Ynorm), torch.tensor(Ymean), torch.tensor(R_np)

def cost_func(X, W, b, Y, R, lambda_):
    y_hat = torch.matmul(X, W.t()) + b
    j = (y_hat - Y) * R
    J = 0.5 * torch.sum(j**2) + lambda_ / 2 * (torch.sum(W**2) + torch.sum(X**2))
    return J

def display_star_rating(rating):
    stars = ""
    for i in range(5):
        if i < rating:
            stars += "⭐"
        else:
            stars += "☆"
    return stars

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'ratings_input' not in st.session_state:
    st.session_state.ratings_input = {}
if 'sample_movies' not in st.session_state:
    st.session_state.sample_movies = None

# Load data
try:
    movies, ratings = load_data()
    
    # Build rating matrix
    Y_df = create_dataset(ratings)
    R_df = (Y_df != 0).astype(int)
    
    # Filter movies that exist in Y_df
    valid_movie_ids = set(Y_df.index)
    filtered_movies = movies[movies['movieId'].isin(valid_movie_ids)]
    
except Exception as e:
    st.error("❌ Error loading data files. Please make sure 'data/movies.csv' and 'data/ratings.csv' exist.")
    st.stop()

# Main header
st.markdown('<h1 class="main-header">🎬 Movie Recommendations</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Rate movies to get personalized recommendations</p>', unsafe_allow_html=True)

# Step 0: Welcome screen
if st.session_state.current_step == 0:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 🌟 How it works:
        1. **Rate some movies** you've watched
        2. **Let our AI analyze** your preferences  
        3. **Get personalized recommendations** tailored just for you
        
        Ready to find your next movie obsession?
        """)
        
        if st.button("🚀 Start Rating Movies", key="start_btn"):
            # Sample 15 popular movies for rating
            st.session_state.sample_movies = filtered_movies.sample(15).reset_index(drop=True)
            st.session_state.current_step = 1
            st.rerun()

# Step 1: Rating interface
elif st.session_state.current_step == 1:
    st.markdown("### 🎯 Rate these movies (0 = Haven't watched, 1-5 = Your rating)")
    
    # Progress bar
    rated_count = sum(1 for rating in st.session_state.ratings_input.values() if rating > 0)
    progress = rated_count / len(st.session_state.sample_movies)
    st.progress(progress)
    st.write(f"📊 Progress: {rated_count}/{len(st.session_state.sample_movies)} movies rated")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    for idx, (_, row) in enumerate(st.session_state.sample_movies.iterrows()):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{row['title']}</div>
                <div class="movie-genre">{getattr(row, 'genres', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Rating buttons
            cols = st.columns([1, 1, 1, 1, 1, 1])
            current_rating = st.session_state.ratings_input.get(row['movieId'], 0)
            
            rating_labels = ["Skip", "★", "★★", "★★★", "★★★★", "★★★★★"]
            
            for i, col in enumerate(cols):
                with col:
                    button_type = "primary" if current_rating == i else "secondary"
                    if st.button(rating_labels[i], key=f"btn_{row['movieId']}_{i}", 
                               type=button_type, use_container_width=True):
                        st.session_state.ratings_input[row['movieId']] = i
                        st.rerun()
            
            # Show current rating
            if current_rating > 0:
                st.markdown(f'<div class="current-rating">Your rating: {rating_labels[current_rating]} ({current_rating}/5)</div>', 
                          unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔄 Get Different Movies"):
            st.session_state.sample_movies = filtered_movies.sample(15).reset_index(drop=True)
            st.session_state.ratings_input = {}
            st.rerun()
    
    with col2:
        if st.button("🏠 Start Over"):
            st.session_state.current_step = 0
            st.session_state.ratings_input = {}
            st.session_state.sample_movies = None
            st.rerun()
    
    with col3:
        if rated_count >= 3:  # Require at least 3 ratings
            if st.button("✨ Get My Recommendations"):
                st.session_state.current_step = 2
                st.rerun()
        else:
            st.button("✨ Get Recommendations", disabled=True, 
                     help="Please rate at least 3 movies to get recommendations")

# Step 2: Generate recommendations
elif st.session_state.current_step == 2:
    st.markdown("### 🤖 Generating Your Personalized Recommendations...")
    
    # Show loading animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Prepare user ratings
        my_ratings = np.zeros(Y_df.shape[0])
        for movie_id, rating in st.session_state.ratings_input.items():
            if rating > 0 and movie_id in Y_df.index:
                row_idx = Y_df.index.get_loc(movie_id)
                my_ratings[row_idx] = rating
        
        status_text.text("🔧 Initializing AI model...")
        progress_bar.progress(0.2)
        
        # Model initialization
        num_features = 50
        num_movies = Y_df.shape[0]
        num_users = Y_df.shape[1] + 1
        
        X = torch.randn((num_movies, num_features), dtype=torch.float64, requires_grad=True)
        W = torch.randn((num_users, num_features), dtype=torch.float64, requires_grad=True)
        b = torch.randn(1, num_users, dtype=torch.float64, requires_grad=True)
        
        Ynorm, Ymean, R = prepare_data(Y_df, R_df, my_ratings)
        opt = optim.Adam([X, W, b], lr=1e-1)
        
        status_text.text("🧠 Training AI on your preferences...")
        progress_bar.progress(0.4)
        
        # Training loop
        for iter in range(50):
            opt.zero_grad()
            cost = cost_func(X, W, b, Ynorm, R, lambda_=1)
            cost.backward()
            opt.step()
            
            if iter % 10 == 0:
                progress_bar.progress(0.4 + (iter/50) * 0.4)
        
        status_text.text("🎯 Generating recommendations...")
        progress_bar.progress(0.9)
        
        # Generate predictions
        pred = torch.matmul(X, W.t()) + b
        predm = pred + Ymean
        my_pred = predm[:, -1]
        idx = torch.argsort(my_pred, descending=True)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Recommendations ready!")
        
        st.markdown("### 🌟 Your Personalized Movie Recommendations")
        
        # Display user's ratings first
        with st.expander("📝 Your Ratings", expanded=False):
            rated_movies = []
            for movie_id, rating in st.session_state.ratings_input.items():
                if rating > 0:
                    title = movies[movies["movieId"] == movie_id]["title"].values
                    if len(title) > 0:
                        rated_movies.append((title[0], rating))
            
            for title, rating in rated_movies:
                st.write(f"• **{title}**: {'★' * rating} ({rating}/5)")
        
        # Show recommendations
        st.markdown("### 🎬 Recommended Movies:")
        
        count = 0
        for i in idx:
            if my_ratings[i] == 0 and count < 10:
                movie_id = Y_df.index[i.item()]
                title = movies[movies["movieId"] == movie_id]["title"].values
                genres = movies[movies["movieId"] == movie_id]["genres"].values
                
                if len(title) > 0:
                    genre_text = genres[0] if len(genres) > 0 else "Unknown"
                    predicted_rating = my_pred[i].item()
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="rec-title">{title[0]}</div>
                        <div class="rec-rating">Predicted Rating: {predicted_rating:.1f}/5.0</div>
                        <div class="rec-genre">Genres: {genre_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    count += 1
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rate More Movies"):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            if st.button("🏠 Start Fresh"):
                st.session_state.current_step = 0
                st.session_state.ratings_input = {}
                st.session_state.sample_movies = None
                st.rerun()
                
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        if st.button("🔙 Go Back"):
            st.session_state.current_step = 1
            st.rerun()

# Sidebar with info
with st.sidebar:
    st.markdown("### 📊 About CineMatch AI")
    st.markdown("""
    This recommendation system uses **collaborative filtering** 
    with machine learning to understand your movie preferences 
    and suggest films you're likely to enjoy.
    
    **How it works:**
    - Analyzes patterns in movie ratings
    - Learns your unique taste profile
    - Matches you with similar users
    - Recommends highly-rated movies from your taste cluster
    
    **Tips for better recommendations:**
    - Rate movies honestly
    - Include different genres you like
    - Rate at least 5-10 movies for best results
    """)
    
    st.markdown("---")
    st.markdown("Made using Streamlit & PyTorch")