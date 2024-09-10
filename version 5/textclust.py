from base import BaseModel
from peewee import fn
import numpy as np
from numpy.linalg import norm as np_norm
from itertools import combinations
from river import feature_extraction
from river import compose
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class TextClust(BaseModel):
    min_distances = []
    
    def __init__(self, conn, r_threshold=0.1, auto_c=None, **kwargs):
        super().__init__(conn, **kwargs)
        self.r_threshold = r_threshold
        self.auto_c = auto_c
    
    def predict_one(self, n_grams, tweet_id, timestep=None, realtime=None):
        return super()._learn_one(n_grams, tweet_id, timestep=timestep, realtime=realtime)
    

_stopwords = set(stopwords.words('english'))

def _clean_text(text):
    text = re.sub(r'http\S+', '', text)
    tokens = word_tokenize(text)
    clean_tokens = [token.lower() for token in tokens if token.lower() not in _stopwords and token.isalnum()]
    clean_text = ' '.join(clean_tokens)
    return clean_text

import psycopg2


hostname = "localhost"
port = 5432
database = "Textclust"
username = "soumitri"
pwd = "1234"
conn = None
cur = None

file_path = '/Users/soumitri/Desktop/Projects/Textclust/TextClust/test_files/climate_data_with_id.json'

conn = psycopg2.connect(
    host = hostname,
    user = username,
    dbname = database,
    password = pwd,
    port = port
)

class CustomPipeline:
    def __init__(self, feature_extractor, predictor):
        self.feature_extractor = feature_extractor
        self.predictor = predictor

    def process(self, data):
        tweet = data.get("tweet", "")
        tweet_id = data.get("tweet_id", None)
        transformed_text = self.feature_extractor.BagOfWords(on='tweet',lowercase=True, ngram_range=(1, 1), stop_words=_stopwords)
        x = transformed_text.transform_one({'tweet': tweet})
        prediction = self.predictor.predict_one(x, tweet_id)
        return prediction

feature_extractor = feature_extraction
predictor = TextClust(conn)

_model = CustomPipeline(feature_extractor, predictor)
print("connected to database..")

climate_data = [] 
with open(file_path, 'r') as file:
    loaded_data = json.load(file)

new_item = []

for data in tqdm(loaded_data[:1200]):
    value = data.get("tweets", None)
    tweet_id = data.get("tweet_id", None)
    if value and value.strip():
        cleaned_tweet = _clean_text(value)
    new_item.append({
        "tweet_id": tweet_id,
        "tweet": cleaned_tweet
    })
for data in tqdm(new_item):
    _model.process(data)

if conn is not None:
    conn.close()
    
    