from fastapi import FastAPI, HTTPException
import pickle
from fastapi.middleware.cors import CORSMiddleware
from .models import recommend_articles, recommend_articles_using_prompt

app = FastAPI()

allowed_origins = [
    # "http://localhost:80", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('objects.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

@app.get('/')
def home():
    return "Yo"

@app.get('/reccomend_with_url/')
def reccomend_with_url(url: str):

    reccomendations = recommend_articles(url, loaded_data['pca_sim_scores'], loaded_data['vectorizer'], loaded_data['pca'], loaded_data['pca_matrix'], loaded_data['df'], loaded_data['url_to_id'])

    return reccomendations

@app.get('/reccomend_with_prompt/')
def reccomend_with_prompt(prompt: str):

    reccomendations = recommend_articles_using_prompt(prompt, loaded_data['vectorizer'], loaded_data['tfidf_matrix'], loaded_data['df'])

    return reccomendations
