import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from .data_transformation import expand_words, clean_data
import requests
from bs4 import BeautifulSoup

def scrape_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        content = ''.join([para.text for para in soup.find(id="article-content").find_all("p")]) if soup.find(id="article-content") else "Content not found"
        return content
    else:
        return {"url": url, "title": f"Failed to retrieve (Status: {response.status_code})"}

def recommend_articles(url, sim_scores, vectorizer, pca,  matrix, df, url_to_id, top_n=10):

    if url not in url_to_id.index:
        content = scrape_data(url)
        content = clean_data(expand_words(content))
        content_vector = vectorizer.transform([content])
        content_vector = pca.transform(content_vector.toarray())
        print(content_vector.shape)
        print(sim_scores.shape)
        sim_scores = cosine_similarity(content_vector, matrix).flatten()
        top_indices = sim_scores.argsort()[-top_n:][::-1]  

    else:
        idx = url_to_id.loc[url]['index']
        sim_scores = list(enumerate(sim_scores[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        top_indices = [i[0] for i in sim_scores]

    recommended_articles = df.iloc[top_indices].reset_index(drop=True)[['url', 'title', 'headline', 'image']]

    return recommended_articles.T.to_dict()

def recommend_articles_using_prompt(prompt, vectorizer, matrix, df, top_n=10):

    prompt = clean_data(expand_words(prompt))
    user_prompt_vector = vectorizer.transform([prompt])
    sim = cosine_similarity(user_prompt_vector, matrix).flatten() 
    top_indices = sim.argsort()[-top_n:][::-1]  
    recommended_articles = df.iloc[top_indices].reset_index(drop=True)[['url', 'title', 'headline', 'image']]
    return recommended_articles.T.to_dict()