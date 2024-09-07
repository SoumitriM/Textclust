#from .db import DATABASE
# from .db.db_models import *
from base_new2 import BaseModel
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
    
    def predict_one(self, n_grams, timestep=None, realtime=None):
        return super()._learn_one(n_grams, timestep=timestep, realtime=realtime)
    
    def init_database(self):
        DATABASE.drop_tables([MicroCluster, Term, TermFrequency])
        DATABASE.create_tables([MicroCluster, Term, TermFrequency])
    
    def initial_merge(self, microclusters, new_cluster):
        distances = self.computeDistances(microclusters, new_cluster)
        min_dist, arg_min_dist = np.min(distances), np.argmin(distances)
        
        ## Check if we shall use AutoR
        if self.auto_c is not None:
            mean, std = np.nanmean(distances), np.nanstd(distances)
            r_threshold = mean - self.auto_c * std
        else:
            r_threshold = self.r_threshold
        
        ## Provide nearest cluster
        if (min_dist < r_threshold):
            self.min_distances.append(min_dist)
            return arg_min_dist
        return None
    
    def cleanup(self, timestep, ignore_ids=[]):
        ## First, we fade all (!) clusters. This may take time
        microclusters = self.fadeClusters(timestep, ignore_ids, greedy=False)
        
        ## Second, reorder based on weight. largest cluster last
        microclusters = sorted(microclusters, key=lambda x: -1. * x.get_weight())
        
        ## Third, compute distance matrix of microclusters
        distances = self.computeDistances(microclusters)
        min_distance = np.nanmean(self.min_distances) if len(self.min_distances) >= 1 else self.r_threshold
        distances = distances < min_distance
        
        ## Fourth, select closest distance and merge if its below threshold
        ## Or delete cluster if it has less than min_terms
        for i in range(len(microclusters)):
            if microclusters[i].merged_with is not None:
                continue
            weight = microclusters[i].get_weight() #weight is None or 
            if weight < (2. ** (-self.fading_factor * self.t_gap)): 
                microclusters[i].soft_delete()
                continue
            
            for k, to_merge in enumerate(distances[i]):
                if to_merge and microclusters[k].merged_with is None:
                    microclusters[i] |= microclusters[k]
                    
            microclusters[i].save()
         
        ## Fifth, cleanup! 
        self.last_cleanup = timestep
        self.min_distances = []
        return None
        
    
    def computeDistances(self, microclusters, other=None):
        ## Compute IDF of based on all TFs
        idf = self.selectIDF()
        
        ## Compute TF-IDFs of microclusters
        tfs = self.selectAllTFIDF(microclusters, idf)
        
        ## If other is not None, we want to compute a vector, otherwise a matrix
        if other is None:
            tfs_idx = combinations(range(len(tfs)), 2)
            distances = np.eye(len(tfs))
        else:
            tfs_new = self.selectTFIDF(other, idf)
            tfs_idx = [(i, None) for i in range(len(tfs))]
            distances = np.zeros(len(tfs))
        
        ## Compute all distances        
        for idx, idy in tfs_idx:
            first = tfs[idx]
            second = tfs[idy] if idy is not None else tfs_new
            
            dist = self.cosine_distance(first, second)
            if idy is not None:
                distances[idx, idy] = dist
                distances[idy, idx] = dist
            else:
                distances[idx] = dist
        return distances
        
        
    def selectAllTFIDF(self, microclusters, idf):
        mc_ids = [mc.id for mc in microclusters]
        idf_keys = list(idf.keys())
        query = TermFrequency.select(TermFrequency.term,
                                     TermFrequency.mc,
                                     fn.sum(TermFrequency.weight).alias('st'))\
                             .where((TermFrequency.mc << mc_ids) & (TermFrequency.term << idf_keys))\
                             .group_by(TermFrequency.term)\
                             .execute()
        
        tfs = []
        for mc_id in mc_ids:
            results = {d.term_id: d.st for d in query if d.mc == mc_id}
            results = np.array([results[k] * v if k in results else 0. for k,v in idf.items()])
            tfs += [results]
        return tfs
    
    def selectIDF(self):#.where(TermFrequency.mc_current.is_null(False))\
        query = TermFrequency.select(TermFrequency.term_id, 
                                     MicroCluster.merged_with, 
                                     fn.COUNT(TermFrequency.term_id).alias('ct'))\
                             .join(MicroCluster)\
                             .where(MicroCluster.merged_with.is_null())\
                             .switch(TermFrequency)\
                             .group_by(TermFrequency.term_id)\
                             .execute()
        N = len(query)
        return {d.term_id: np.log(N / d.ct) for d in query if d.ct > 0}
    
    def selectTFIDF(self, microcluster, idf):
        idf_keys = list(idf.keys())
        query = TermFrequency.select(TermFrequency.term,
                                     fn.sum(TermFrequency.weight).alias('st'))\
                             .where((TermFrequency.mc == microcluster.id) & (TermFrequency.term << idf_keys))\
                             .group_by(TermFrequency.term)
        results = {d.term_id: d.st for d in query}
        return np.array([results[k] * v if k in results else 0. for k,v in idf.items()]) # TODO Check if zero is right here!
    
    def cosine_distance(self, tfidf1, tfidf2):
        return 1. - (tfidf1 @ tfidf2) / (np_norm(tfidf1) * np_norm(tfidf2) + 1e-12)
 
 
 
   
#tx = TextClust()


item2 =  [{'climate': 1, 'change': 1, 'is': 1, 'a': 1, 'pressing': 1, 'global': 1, 'issue': 1},
{"have":1, "a":2},
{"hello": 1, "here": 3, "test": 1},
{"Lord":1, "Sita": 2},
{'climate': 1, 'global': 1, 'not':1, 'issue': 1},
{"Lord":1, "Sita": 2},
{"random": 1, "words": 2, "there": 4},
{"random": 2, "love": 1, "words": 2},
{"i": 1, "cat": 1, "dog": 2}
]
_stopwords = set(stopwords.words('english'))
# x = feature_extraction.BagOfWords(lowercase=True, ngram_range=(1), stop_words=_stopwords)
#start connection
# _model = compose.Pipeline(
#             feature_extraction.BagOfWords(lowercase=True, ngram_range=(1,1), stop_words=_stopwords),
#             TextClust()
#         )

def _clean_text(text):
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Tokenize text
            tokens = word_tokenize(text)
            # Remove stopwords and noise tokens
            clean_tokens = [token.lower() for token in tokens if token.lower() not in _stopwords and token.isalnum()]
            # Reconstruct text
            clean_text = ' '.join(clean_tokens)
            return clean_text
climate_data = [
    {'text': "Climate change is real."},
    {'text': "Human activities are the primary cause of climate change."},
    {'text': "Greenhouse gas emissions are rising at an alarming rate."},
    {'text': "The polar ice caps are melting due to global warming."},
    {'text': "Sea levels are rising, threatening coastal communities."},
    {'text': "Extreme weather events are becoming more frequent and severe."},
    {'text': "Climate change is affecting biodiversity and ecosystems."},
    {'text': "Deforestation contributes significantly to climate change."},
    {'text': "Renewable energy sources can help mitigate climate change."},
    {'text': "Climate change poses a serious threat to food security."},
    {'text': "Ocean acidification is a direct result of increased CO2 levels."},
    {'text': "Climate change disproportionately impacts vulnerable populations."},
    {'text': "Sustainable practices can reduce the impact of climate change."},
    {'text': "Global cooperation is essential to combat climate change effectively."},
    {'text': "The Paris Agreement aims to limit global warming to below 2°C."},
    {'text': "Individual actions can contribute to reducing carbon footprints."},
    {'text': "Climate change affects human health in various ways."},
    {'text': "The transition to a low-carbon economy is crucial."},
    {'text': "Educating others about climate change is vital for future generations."},
    {'text': "Immediate action is needed to address the climate crisis."}
]

# with open('/Users/soumitri/Desktop/Projects/Textclust/TextClust/rbook.json', 'r') as file:
#     climate_data = json.load(file)

# new_item = []
# for data in tqdm(climate_data):
#     value = data['text']
#     new_item.append(_clean_text(value))
#     #_model.predict_one(value) 
# # print(1)
# for data in tqdm(new_item[:5000]):
#     _model.predict_one(data)
# # for item in item2:
# #     tx.predict_one(item)
# # print(1)

import psycopg2


hostname = "localhost"
port = 5432
database = "Textclust"
username = "soumitri"
pwd = "1234"
conn = None
cur = None

# try: 
#     conn = psycopg2.connect(
#         host = hostname,
#         user = username,
#         dbname = database,
#         password = pwd,
#         port = port
#     )
#     cur = conn.cursor()
#     _model = compose.Pipeline(
#             feature_extraction.BagOfWords(lowercase=True, ngram_range=(1,1), stop_words=_stopwords),
#             TextClust(cur)
#         )
#     print("connected..")
#     with open('/Users/soumitri/Desktop/Projects/Textclust/TextClust/rbook.json', 'r') as file:
#         climate_data = json.load(file)
#     new_item = []
#     for data in tqdm(climate_data):
#         value = data['text']
#         new_item.append(_clean_text(value))

#     for data in tqdm(new_item[:2000]):
#         _model.predict_one(data)
# except Exception as error:
#     print(error)
# finally:
#     # Close the connection in the 'finally' block to ensure it happens after processing
#     if conn is not None:
#         conn.close()
#         print("disconnected..")
#     if conn is not None:
#         conn.close()
#         print("disconnected..")

conn = psycopg2.connect(
    host = hostname,
    user = username,
    dbname = database,
    password = pwd,
    port = port
)
#cur = conn.cursor()
_model = compose.Pipeline(
        feature_extraction.BagOfWords(lowercase=True, ngram_range=(1,1), stop_words=_stopwords),
        TextClust(conn)
    )
print("connected..")

climate_data = []
with open('/Users/soumitri/Desktop/Projects/Textclust/TextClust/original.json', 'r') as file:
    for line in file:
        try:
            json_object = json.loads(line)
            climate_data.append(json_object)
            #print(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")


new_item = []

for data in tqdm(climate_data):
    value = data.get("tweets", None)  # Use .get() to avoid KeyError
    if value and value.strip():
        new_item.append(_clean_text(value))



for data in tqdm(new_item[:20000]):
    _model.predict_one(data)

if conn is not None:
    conn.close()