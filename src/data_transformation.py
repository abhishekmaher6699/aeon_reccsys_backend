import pandas as pd
import re 
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import logging

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def expand_words(text):
    return contractions.fix(text)

def remove_punctuations(text):
    spaced_punc = "!#$%‘&\()*+,-./:;<=>?@[\\]^_`{|}~…"
    non_spaced_punc = "’\'\""

    text = text.translate(str.maketrans(spaced_punc, ' ' * len(spaced_punc)))
    text = text.translate(str.maketrans('', '', non_spaced_punc))

    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)    
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    return ' '.join(cleaned_tokens)

def clean_data(text):
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    
    return text.lower()

def main():
    try:
        df = pd.read_csv("./artifacts/data.csv")
        df['content'] =  df['content'].apply(expand_words)
        logging.info("Expanded contradictions")
        df['transformed_content'] = df['content'].apply(clean_data)
        logging.info("Transformed data")

        df.to_csv("./artifacts/data.csv", index=False)

        logging.info("Saved data")

    except Exception as e:
        logging.info(e)

if __name__ == "main":
    main()