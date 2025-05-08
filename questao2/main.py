import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix
import time
import psutil

#alunos: gustavo schneider rodrigues, felipe beppler huller

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory used: {process.memory_info().rss / (1024 ** 2):.2f} MB")

print("Loading data...")
start_time_total = time.time()
ratings = pd.read_csv('./questao2/ratings.csv')
movies = pd.read_csv('./questao2/movies.csv')

movies['genres_list'] = movies['genres'].str.split('|')
genre_map = movies.explode('genres_list').groupby('title')['genres_list'].apply(list).to_dict()

print_memory_usage()
print(f"Data loaded. Time: {time.time() - start_time_total:.2f}s")

print("\nFiltering data...")
start_time = time.time()

ratings['watched'] = ratings['rating'] >= 3.5
user_movie_matrix = ratings[ratings['watched']][['userId', 'movieId']]

top_movies = ratings['movieId'].value_counts().head(200).index.tolist()
filtered_data = user_movie_matrix[user_movie_matrix['movieId'].isin(top_movies)]

active_users = filtered_data['userId'].value_counts()[lambda x: x >= 10].index.tolist()[:100]
filtered_data = filtered_data[filtered_data['userId'].isin(active_users)]

print_memory_usage()
print(f"Data filtered. Time: {time.time() - start_time:.2f}s")

print("\nCreating transactions...")
start_time = time.time()

movie_title_map = movies.set_index('movieId')['title'].to_dict()
transactions = []
for user_id, group in filtered_data.groupby('userId'):
    user_movies = [movie_title_map[movie_id] for movie_id in group['movieId'] 
                  if movie_id in movie_title_map]
    if len(user_movies) >= 2:
        transactions.append(user_movies)

print_memory_usage()
print(f"Transactions created. Time: {time.time() - start_time:.2f}s")

print("\nEncoding transactions...")
start_time = time.time()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
sparse_matrix = csr_matrix(te_ary.astype("bool"))
df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=te.columns_)

print_memory_usage()
print(f"Transactions encoded. Time: {time.time() - start_time:.2f}s")

print("\nRunning FP-Growth...")
start_time = time.time()

frequent_itemsets = fpgrowth(
    df,
    min_support=0.1,
    use_colnames=True,
    max_len=2
)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

print_memory_usage()
print(f"FP-Growth completed. Time: {time.time() - start_time:.2f}s")

def recommend_movies(watched_movies, rules_df, top_n=5):
    watched_genres = set()
    for movie in watched_movies:
        watched_genres.update(genre_map.get(movie, []))
    
    recommendations = {}
    watched_set = set(watched_movies)
    
    for _, rule in rules_df.iterrows():
        if rule['antecedents'].issubset(watched_set):
            for movie in rule['consequents'] - watched_set:
                score = rule['confidence'] * rule['lift'] * rule['support']
                
                movie_genres = set(genre_map.get(movie, []))
                genre_match = len(watched_genres & movie_genres) / len(watched_genres) if watched_genres else 0
                score *= (1 + genre_match)
                
                recommendations[movie] = max(recommendations.get(movie, 0), score)
    
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

if not rules.empty:
    print("\n=== RECOMMENDATION EXAMPLES ===")
    
    action_fan = ["Dark Knight, The (2008)", "Inception (2010)", "Matrix, The (1999)"]
    valid_action = [m for m in action_fan if m in te.columns_]
    
    if valid_action:
        print("\n🎬 Action/Sci-Fi Fan (watched: {})".format(", ".join(valid_action)))
        recs = recommend_movies(valid_action, rules)
        for i, (movie, score) in enumerate(recs, 1):
            print(f"{i}. {movie} (score: {score:.3f})")
            print(f"   Genres: {', '.join(genre_map.get(movie, ['Unknown']))}")
    
    romance_fan = ["Pretty Woman (1990)", "When Harry Met Sally... (1989)", "Notting Hill (1999)"]
    valid_romance = [m for m in romance_fan if m in te.columns_]
    
    if valid_romance:
        print("\n💖 Romance/Comedy Fan (watched: {})".format(", ".join(valid_romance)))
        recs = recommend_movies(valid_romance, rules)
        for i, (movie, score) in enumerate(recs, 1):
            print(f"{i}. {movie} (score: {score:.3f})")
            print(f"   Genres: {', '.join(genre_map.get(movie, ['Unknown']))}")

total_time = time.time() - start_time_total
print("\n" + "="*50)
print(f"FINAL REPORT")
print(f"Total time: {total_time:.2f}s")
print(f"Peak memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
print("="*50)