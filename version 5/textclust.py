from .db import DATABASE
from .db.db_models import *
from .base import BaseModel
from peewee import fn
import numpy as np
from numpy.linalg import norm as np_norm
from itertools import combinations

class TextClust(BaseModel):
    min_distances = []
    
    def __init__(self, r_threshold=0.1, auto_c=None, **kwargs):
        super().__init__(**kwargs)
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
            if weight < (2. ** (-self.fading_factor * self.t_gab)): 
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