import streamlit as st
import gc
import pandas as pd
import subprocess
import sys

# Ensure scikit-learn is installed
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    st.warning("scikit-learn not found. Attempting to install scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import hstack, vstack

st.title("Chunk-Based Similarity Example")

# Use a chunk size for processing
chunk_size = 200

# Load the CSV from the GitHub raw URL
csv_url = "https://raw.githubusercontent.com/Tfields77/Wine-Recommendation/main/winemag-data-130k-v2.csv"
df = pd.read_csv(csv_url)

# Sample the dataframe to reduce its size
df = df.sample(frac=0.3, random_state=42)
total_rows = df.shape[0]
st.write("Total rows after sampling:", total_rows)

# Lists for storing feature matrices and metadata
X_list = []  
meta_list = []  

# Use a sample fraction for fitting the transformers
sample_frac = 0.05  
sample_variety = df[['variety']].sample(frac=sample_frac, random_state=42)
sample_region = df[['region_1']].sample(frac=sample_frac, random_state=42)
sample_description = df['description'].sample(frac=sample_frac, random_state=42)

pre_encoder_variety = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
pre_encoder_variety.fit(sample_variety)

pre_encoder_region = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
pre_encoder_region.fit(sample_region)

pre_tfidf = TfidfVectorizer(stop_words='english', max_features=500)
pre_tfidf.fit(sample_description)

def process_and_calculate_similarity(start_index):
    # Load chunk rows
    chunk = df.iloc[start_index:start_index + chunk_size]

    # Transform the features
    variety_encoded = pre_encoder_variety.transform(chunk[['variety']])
    region_encoded = pre_encoder_region.transform(chunk[['region_1']])
    description_tfidf = pre_tfidf.transform(chunk['description'])
    
    # Combine features
    X = hstack([variety_encoded, region_encoded, description_tfidf])
    similarity_matrix = cosine_similarity(X)
    
    X_list.append(X)
    meta_list.append(
        chunk[['title', 'description', 'taster_twitter_handle', 'variety', 'price', 'region_1']]
    )
    
    # Return the similarity matrix and the current chunk
    return similarity_matrix, chunk  

# Loop through dataset in chunks and compute recommendations
recommendations = []
current_chunk = 0

while current_chunk * chunk_size < total_rows:
    similarity_matrix, chunk = process_and_calculate_similarity(current_chunk * chunk_size)
    
    # For each wine in the chunk, find its top similar wines
    for wine_index in range(chunk.shape[0]):
        similar_wines = similarity_matrix[wine_index]
        similar_wine_indices = similar_wines.argsort()[-6:-1][::-1]
        recommended_wines = chunk.iloc[similar_wine_indices]
        recommendations.append(recommended_wines[['title', 'variety', 'price', 'region_1']])
    
    current_chunk += 1
    del similarity_matrix, chunk
    gc.collect()

final_recommendations = pd.concat(recommendations, ignore_index=True)
st.write("Final Recommendations:")
st.dataframe(final_recommendations)

X_full = vstack(X_list)
meta_full = pd.concat(meta_list, ignore_index=True)

st.write("X_full shape:", X_full.shape)
st.write("meta_full shape:", meta_full.shape)
