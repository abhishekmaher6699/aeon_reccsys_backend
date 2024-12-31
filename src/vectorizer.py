import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# import logging
import pickle

def fetch_data():

    try:
        df = pd.read_csv("./artifacts/data.csv")

        print("Data fetched successfully")

        return df
    except Exception as e:
        print(e)

def vectorize(data):

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix  = vectorizer.fit_transform(data)

        print("Tfidf matrix built")

        pca = PCA(n_components=100)
        pca_matrix = pca.fit_transform(tfidf_matrix.toarray())

        print("PCA matrix built!")

        return vectorizer, tfidf_matrix, pca, pca_matrix
    
    except Exception as e:
        print(e)

def generate_sim_matrix(pca_matrix, tfidf_matrix):

    try:
        pca_sim_scores = cosine_similarity(pca_matrix, pca_matrix)
        tf_sim_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return pca_sim_scores, tf_sim_scores
    
    except Exception as e:
        print(e)

def main():
    try:
        df = fetch_data()

        url_to_id = df['url'].reset_index().set_index("url")
        
        vectorizer, tfidf_matrix, pca, pca_matrix = vectorize(df['transformed_content'])

        pca_sim_scores, tf_sim_scores = generate_sim_matrix(pca_matrix, tfidf_matrix)

        data_to_pickle = {
            "df": df,
            "url_to_id": url_to_id,
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "pca": pca,
            "pca_matrix": pca_matrix,
            "pca_sim_scores": pca_sim_scores,
            # "tf_sim_scores": tf_sim_scores
        }

        with open('./artifacts/objects.pkl', 'wb') as f:
            pickle.dump(data_to_pickle, f)

        print("All objects pickled successfully!")
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()








