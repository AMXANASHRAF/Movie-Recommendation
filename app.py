import streamlit as st
import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.title("🎬 Movie Recommendation System")

# Cache the data loading and preprocessing to improve performance
@st.cache_data
def load_and_process_data():
    try:
        # Load the data - update these paths to your actual file locations
        movie = pd.read_csv(r"C:\Users\amaan\Downloads\movies.csv")  # Update path
        credits = pd.read_csv(r"C:\Users\amaan\Downloads\credits.csv")  # Update path
        
        # Merge datasets
        movies = movie.merge(credits, on='title')
        movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
        movies.dropna(inplace=True)
        
        return movies
    except FileNotFoundError:
        st.error("CSV files not found. Please ensure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Data preprocessing functions
def convert(obj):
    """Convert JSON string to list of names"""
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

def convert_cast(obj):
    """Convert cast JSON to list of top 3 actor names"""
    try:
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    except:
        return []

def fetch_director(obj):
    """Extract director name from crew JSON"""
    try:
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    except:
        return []

@st.cache_data
def preprocess_data(movies):
    """Preprocess the movie data"""
    # Apply conversion functions
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Clean overview text
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    # Remove spaces from names
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Combine all features into tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create final dataframe
    new_df = movies[['movie_id','title','tags']].copy()
    
    # Convert tags to string
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    
    # Apply stemming
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    
    new_df['tags'] = new_df['tags'].apply(stem)
    
    return new_df

@st.cache_data
def create_similarity_matrix(new_df):
    """Create similarity matrix from processed data"""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

# Recommendation function
def recommend(movie, new_df, similarity):
    """Get movie recommendations"""
    try:
        if movie not in new_df['title'].values:
            return []
        
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommendations = []
        for i in movies_list:
            recommendations.append(new_df.iloc[i[0]].title)
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# Main app logic
def main():
    # Load and process data
    movies = load_and_process_data()
    
    if movies is not None:
        with st.spinner("Processing movie data..."):
            new_df = preprocess_data(movies)
            similarity = create_similarity_matrix(new_df)
        
        st.success(f"Loaded {len(new_df)} movies successfully!")
        
        # Movie selection
        selected_movie = st.selectbox(
            "Select a movie to get recommendations:",
            options=sorted(new_df['title'].values),
            index=0
        )
        
        # Show selected movie info
        if selected_movie:
            movie_info = new_df[new_df['title'] == selected_movie]
            if not movie_info.empty:
                st.write(f"**Selected Movie:** {selected_movie}")
                st.write(f"**Movie ID:** {movie_info.iloc[0]['movie_id']}")
        
        # Get recommendations
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Finding similar movies..."):
                recommendations = recommend(selected_movie, new_df, similarity)
            
            if recommendations:
                st.success("Here are some movies you might like:")
                
                # Display recommendations in a nice format
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {rec}")
            else:
                st.error("No recommendations found. Please try another movie.")
    else:
        st.error("Failed to load movie data. Please check your CSV files.")

if __name__ == "__main__":
    main()



# %%



