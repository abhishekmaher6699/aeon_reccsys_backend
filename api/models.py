import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from .data_transformation import expand_words, clean_data

def recommend_articles(url, sim_scores, df, url_to_id, top_n=10):
    
    idx = url_to_id.loc[url]['index']
    sim_scores = list(enumerate(sim_scores[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    top_indices = [i[0] for i in sim_scores]
    recommended_articles = df.iloc[top_indices][['url', 'title', 'content']]
    recommended_articles['content'] = recommended_articles['content'].apply(lambda x: x[:100] if isinstance(x, str) else x)
    return recommended_articles.T.to_dict()


def recommend_articles_using_prompt(prompt, vectorizer, matrix, df, top_n=10):

    prompt = clean_data(expand_words(prompt))
    user_prompt_vector = vectorizer.transform([prompt])
    sim = cosine_similarity(user_prompt_vector, matrix).flatten() 
    top_indices = sim.argsort()[-top_n:][::-1]  
    recommended_articles = df.iloc[top_indices][['url', 'title', 'content']]
    recommended_articles['content'] = recommended_articles['content'].apply(lambda x: x[:100] if isinstance(x, str) else x)
    return recommended_articles.T.to_dict()