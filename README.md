#  Tag-Based Movie Recommendation System (TMDB 5000)

This project is a **content-based movie recommender system** that uses **tag-based similarity** to suggest movies. It utilizes the TMDB 5000 Movies and Credits datasets to process metadata like cast, crew, genres, keywords, and overview to recommend similar movies.

---

##  Dataset Used

- **tmdb_5000_movies.csv**
- **tmdb_5000_credits.csv**

These datasets are publicly available on Kaggle and contain rich metadata for thousands of movies.

---

##  Tech Stack

- **Python 3.x**
- **Pandas** for data preprocessing
- **Scikit-learn** for vectorization and similarity computation
- **Numpy** for numerical computation
- **NLTK** (optional) for text processing

---

##  How It Works

1. **Data Merging**: Combines movie and credit data on the `id` field.
2. **Feature Extraction**: Extracts relevant features:
   - `genres`, `keywords`, `overview`, `cast`, and `crew`
3. **Preprocessing**:
   - JSON decoding of stringified fields
   - Limiting to top 3 actors and the director
   - Creating a unified `tags` column
4. **Vectorization**:
   - The `tags` column is transformed using `CountVectorizer`
5. **Similarity Matrix**:
   - Cosine similarity is used to compute distances between movies
6. **Recommendation Function**:
   - Given a movie title, returns top N similar movies

---






