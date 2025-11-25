![logo](/frontend/assets/images/logo.png)

# 🎬 CineMatch — Movie Recommendation System

CineMatch is an explainable, content-based movie recommendation system built with:

- Python
- TF-IDF vectorization
- Sentence-BERT embeddings (for semantic understanding)
- Collaborative Filtering using MovieLens dataset
- Hybrid model combining content-based + collaborative recommendations
- A small frontend (HTML/CSS/JS)
- TMDB movie dataset

Developed as part of the *Machine Learning Business Projects* course.


## ✨ Features

✔ Content-based recommendation system  
✔ Collaborative Filtering (CF) using MovieLens dataset  
✔ Hybrid recommender (combining content + CF)  
✔ Sentence-BERT embeddings for improved semantic recommendations  
✔ Explainable AI (shared genres, cast, director, similarity score)  
✔ FastAPI backend  
✔ Clean frontend UI with posters  
✔ Auto-suggest / autocomplete on movie titles  
✔ Handles ~30k popular English movies  
✔ Works offline except posters 


## 📁 Project Structure


```
cinematch/
│
├── api/
│ ├── main.py
│ ├── data_loader.py
│ ├── models_cf.py
│ ├── models_content.py
│ ├── vector_index.py
│ ├── requirements.txt
│ └── cache/
│   └── bert_embeddings.npy
│
├── frontend/
│ ├── index.html
│ ├── movie.html
│ ├── styles.css
│ ├── movie.css
│ ├── movie.js
│ ├── app.js
│ └── assets/
│   ├── images/
│   └── logo.png
│
├── eda/
│ ├── notebook.ipynb
│ └── tmdb_movies.csv
│ └── movielens/
│
├── data/
│ └── data.csv
│
└── README.md
```


## 🧠 Machine Learning Approach

### 1. Data Preparation
- Filtered movies (English only)
- Removed missing overviews
- Selected ~30,000 most popular films
- Cleaned text columns (genres, cast, director…)
- Built a combined textual feature:  
  *overview + genres + cast + director + tagline*

### 2. TF-IDF Vectorization
- Vectorizer with `max_features=50,000`
- Learns a weighted vocabulary (trained ML model)

### 3. k-NN Recommender (cosine)
- Finds nearest movies in vector space
- Provides top-N recommendations

### 4. Collaborative Filtering (CF)
- Uses MovieLens dataset (100k ratings) for collaborative filtering
- Trains a user-based or item-based CF model (kNN or matrix factorization)

### 5. Hybrid Model
- Combines the results from the **content-based** and **collaborative** models
- Uses a weighted score for a more personalized recommendation

### 6. Sentence-BERT Embeddings
- Uses `all-mpnet-base-v2` or similar BERT models to compute semantic embeddings
- Improves recommendations by understanding movie content context beyond simple keyword matches

### 7. Explainable AI
Each recommendation includes:
- Shared genres  
- Shared actors (top 5)  
- Same director flag  
- Similarity score (cosine)
This makes the system interpretable and user-friendly.


## 🌐 API Endpoints

### `GET /recommend?title=Inception`
→ Returns recommended movies based on content-based or hybrid model (default).

### `GET /recommend?title=Inception&mode=cf`
→ Returns recommended movies based on collaborative filtering using MovieLens dataset.

### `GET /recommend?title=Inception&mode=bert`
→ Returns recommended movies based on semantic similarity using Sentence-BERT embeddings.

### `GET /explain?source=Inception&candidate=Interstellar`
→ Explains why a recommendation was made, including shared genres, cast, director, and similarity score.

### `GET /suggest?query=ince`
→ Autocomplete suggestions based on movie titles.

### `GET /movie?title=Inception`
→ Returns details of a specific movie including genres, director, cast, tagline, and overview.



## 🚀 Running the Backend

```
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

FastAPI runs at:  
http://127.0.0.1:8000


## 🌈 Running the Frontend

Open:

```
frontend/index.html
```

The UI automatically communicates with the backend.
For movie details, use URLs like `movie.html?title=Inception`.

## 🧪 Dataset

Using the updated 2024-2025 TMDB movies dataset:

- title  
- overview  
- genres  
- cast  
- director  
- popularity  
- poster_path  

Used the subset most relevant for recommendations.


## 📊 Why this is a valid ML Project

- TF-IDF = trained ML model  
- k-NN = trained ML model  
- Collaborative Filtering (CF) = trained ML model  
- Hybrid recommender (combining content-based + CF)  
- Sentence-BERT = trained ML model  
- Clear ML pipeline  
- Explainability (XAI) implemented  
- Real dataset  
- Real full-stack application (API + UI)  


## 🎯 Limitations

- Only content-based + collaborative filtering (no user profiles or ratings yet)
- Autocomplete needed for clean UX  
- Posters depend on TMDB CDN  


## 🛠 Possible Future Improvements

- Add user profiles + ratings  
- Add Sentence-BERT embeddings for more powerful recommendations  
- Fully deploy the app online (Railway/Render)  
- Add a multi-criteria search page (e.g., filter by genre, year, etc.)


## 📌 Credits

Developed as a Machine Learning Business Project (2025).
[Dataset © TMDB.](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates?resource=download)
