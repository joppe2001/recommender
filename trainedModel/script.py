
import pandas as pd
import numpy as np
import pickle
import os

def load_saved_data(directory):
    df = pd.read_pickle(os.path.join(directory, "savedModel/anime_dataframe.pkl"))
    with open(os.path.join(directory, "savedModel/cosine_similarity_matrix.pkl"), 'rb') as f:
        cosine_sim = pickle.load(f)
    return df, cosine_sim

def recommend_anime(df, cosine_sim, user_history, N=10):
    user_anime_indices = []
    for title in user_history:
        matching_anime = df[(df['engName'].str.lower() == title.lower()) | (df['synonymsName'].str.contains(title, case=False, na=False))]
        if matching_anime.empty:
            print(f"Warning: Anime titled '{title}' not found in the dataset.")
        else:
            user_anime_indices.append(matching_anime.index[0])
    
    avg_sim_scores = cosine_sim[user_anime_indices].mean(axis=0)
    max_score = df["score"].max()
    normalized_scores = df["score"].fillna(0) / max_score
    combined_scores = avg_sim_scores + normalized_scores
    top_indices = combined_scores.argsort()[-N-1:-1][::-1]
    recommended_anime = df.iloc[top_indices][['engName', 'score']]
    

    for title in user_history:
        matching_indices = recommended_anime[recommended_anime['engName'].str.contains(title, case=False)].index
        if len(matching_indices) > 2:
            drop_indices = matching_indices[2:]
            recommended_anime.drop(drop_indices, inplace=True)
            
    return recommended_anime

if __name__ == '__main__':
    save_directory = ""
    df, cosine_sim = load_saved_data(save_directory)

    user_history = []
    anime_name = ""
    while anime_name.lower() != "done":
        anime_name = input("Enter the name of an anime you've watched (or 'done' to finish): ")
        if anime_name.lower() != "done":
            user_history.append(anime_name)

    recommendations = recommend_anime(df, cosine_sim, user_history)
    print("\nTop Recommendations:")
    for idx, (name, score) in enumerate(recommendations.values, 1):
        print(f"{idx}. {name} (Score: {score:.2f})")
