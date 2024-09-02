from peewee import fn
from abc import ABC
import numpy as np
from datetime import datetime
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
import uuid
from collections import defaultdict
from memory.models import Weight, TermDictionary, MicroCluster, Term, TermFrequency

class BaseModel():
    start_realtime = None
    last_cleanup = None
    n_observations = 0
    
    
    def __init__(self, fading_factor=0.0005, t_gap=50, auto_cleanup=True, use_realtime=False):
        self.fading_factor = fading_factor
        self.t_gap = t_gap
        self.fading_const = 2. ** (-fading_factor)
        self.threshold_wt = 2. ** (-self.fading_factor * self.t_gap)
        self.auto_cleanup = auto_cleanup
        self.use_realtime = use_realtime
        self.termDictionary = TermDictionary()
        self.microClusters = {}
        self.tfs = []
        self.mc_df = pd.DataFrame()
        self.mcs = pd.DataFrame(columns=['memory_id', 'time_stamp', 'weight', 'n_observations', 'merged_with'])
        self.d_merge = 0
        self.n_merge = 0
        
    
    def create_new(self, n_grams, timestep):
        return self._create_new_mc(n_grams, timestep)
    
    # def fade_clusters(self, timestep, ignore_ids=[], greedy=False):
    #     return self.fade_clusters(timestep, ignore_ids=ignore_ids, greedy=greedy)
    
    def microclusters(self, ignore_ids=[]):
        return self.microClusters(ignore_ids)
           
    def step(self, n_grams, timestep):
        return self._step(n_grams, timestep)
    
        
    def _learn_one(self, n_grams, timestep=None, realtime=None):
        ## Save starting time and compute timestep if missing
        if self.use_realtime:
            realtime = datetime.now() if realtime is None else realtime
            self.start_realtime = realtime if self.start_realtime is None else self.start_realtime
            timestep = max(realtime - self.start_realtime, 0)
        else:
            timestep = self.n_observations if timestep is None else timestep
        
        ## Increase counter and execute step
        self.n_observations += 1
        return self.step(n_grams, timestep)
        
    def _step(self, n_grams, timestep):
        ## If n_grams is empty, don't make prediction
        if len(n_grams) == 0:
            return None
        ## Create new MicroCluster
        self._create_new_mc(n_grams, timestep)
        if timestep%200 == 0:
          print("mcs...", self.mcs.shape[0])
          self._cleanup()
        # if timestep%70 == 0:
        #   self.merge_clusters_cleanup()
        return 1 # Provide new Cluster ID as prediction
    
    
    def _create_new_mc(self, n_grams, timestep):
        curr_tokens = []
        for token, freq in n_grams.items():
          curr_tokens.append(token)
          term = self.termDictionary.get_term(token)
          if term is None:
              term_obj = Term(token=token, timestep=timestep, weight=1, document_frequency=1)
              self.termDictionary.add_term(term=token, term_obj=term_obj)
          else:
            term.weight.weight += 1
            term.time_stamp = timestep
            term.document_frequency += 1
        memory_id = uuid.uuid4().hex
        #creating / appending to tf dataframe
        new_row_df = pd.DataFrame(n_grams, index=[memory_id])
        self.mc_df = pd.concat([self.mc_df, new_row_df], axis=0).fillna(0)
        #appending new_mc to mcs dataframe
        new_mcrow_df = pd.DataFrame([{'memory_id': memory_id, 'time_stamp':timestep, 'weight': 1, 'n_observations': 1}])
        self.mcs = pd.concat([self.mcs, new_mcrow_df], ignore_index = True)
        if(self.mcs.shape[0] > 2):
          self.fade_clusters(memory_id, timestep)
          self.fade_terms(curr_tokens, timestep)
          self.merge_if_eligible(memory_id)

    
    '''This function calculates the threshold value for minimum distance of merging'''
    def calculate_threshold(self, similarityMatrix, maxSimilarityIndex):
      tr = 0
      smatrix = np.array(1 - similarityMatrix)
      filtered_values = np.delete(smatrix, maxSimilarityIndex)
      mean_excluding_min = np.mean(filtered_values)
      std_excluding_min = np.std(filtered_values)
      # Calculate the desired value: mean - 0.5 * standard deviation
      tr = mean_excluding_min - 0.5 * std_excluding_min
      return tr
            

      
    '''This function merges mc_new to its closest mc which becomes its parent mc'''
    def mergeClusters(self, child_id, parent_id):
      #print("merging..")
      # Get the index of the parent and new rows
      #print(self.mcs.head(20))
      parent_index = self.mcs[self.mcs['memory_id'] == parent_id].index[0]
      new_index = self.mcs[self.mcs['memory_id'] == child_id].index[0]
      # Modify the original DataFrame directly using loc
      self.mcs.loc[parent_index, 'n_observations'] += 1
      self.mcs.loc[parent_index, 'merged_with'] = self.mcs.loc[new_index, 'memory_id']
      self.mcs.loc[parent_index, 'weight'] += self.mcs.loc[new_index, 'weight']
      self.mcs.loc[parent_index, 'time_stamp'] = self.mcs.loc[new_index, 'time_stamp']
      self.mcs.drop(new_index)
      self.mc_df.loc[parent_id] += self.mc_df.loc[child_id]
      self.mc_df.drop(child_id)
      
    
  # Function to calculate cosine similarity
    def calculate_similarity(self, tf_new, tf_matrix, ids):
        cosine_similarities = np.dot(tf_matrix, tf_new)
        min_distance_index = np.argmax(cosine_similarities)
        threshold = self.calculate_threshold(cosine_similarities, min_distance_index)
        return (1 - cosine_similarities[min_distance_index], ids[min_distance_index], threshold)


    def merge_if_eligible(self, child_id):
      tfidf_df = self.mc_df.copy(deep=False)
      for token in tfidf_df.columns:
        doc_freq = self.termDictionary.get_doc_freq(token)
        tfidf_df[token] = tfidf_df[token] / doc_freq
      tfidfm = tfidf_df.values
      tfidfnew = tfidfm[-1]
      tfidfm = np.delete(tfidfm, -1, axis=0)
      row_ids = self.mc_df.index.tolist()
      #print(tfidfnew.shape,tfidfm.shape)
      minDist, parent_id, tr = self.calculate_similarity(tfidfnew,tfidfm, row_ids)
      
      if minDist < tr:
        self.mergeClusters( child_id, parent_id )
        self.n_merge +=1
        self.d_merge += minDist



    def fade_weight(self, timestep, token):
      fading_const = 2. ** (-self.fading_factor)
      time_difference = timestep - token.time_stamp
      new_wt = token.weight.weight * (fading_const ** time_difference)
      return new_wt
      
    '''This function fades tfs after every 50 observations'''
    def fade_terms(self, curr_tokens, timestep):
        for token, term_obj in self.termDictionary.items():
          if token not in curr_tokens:
            new_wt = max(0,term_obj.weight.weight * (2 **(-self.fading_factor * (timestep - term_obj.time_stamp))))
            self.termDictionary[token].weight.weight  = new_wt
            
    
    '''This function fades clusters after every 50 observations'''     
    def fade_clusters(self, memory_id, timestep):
      mask = self.mcs['memory_id'] != memory_id
      self.mcs.loc[mask, 'weight'] = self.mcs.loc[mask].apply(lambda row: max(0, row['weight'] * (2 ** (-self.fading_factor * (timestep - row['time_stamp'])))), axis=1)
      
    def group_similar_clusters(self, dist_m, row_ids, threshold):
      adjacency_matrix = (dist_m < threshold).astype(int)
      sparse_graph = csr_matrix(adjacency_matrix)
      n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
      clusters = [[] for _ in range(n_components)]
      for idx, label in enumerate(labels):
          clusters[label].append(row_ids[idx])
      
      return clusters
    
    def merge_similar_clusters(self, groups):
      for group in groups:
          group_rows = self.mcs[self.mcs['memory_id'].isin(group)]
          largest_timestamp_row = group_rows.loc[group_rows['time_stamp'].idxmax()]
          total_weight = group_rows['weight'].sum()
          self.mcs.loc[largest_timestamp_row.name, 'weight'] = total_weight
          self.mcs = self.mcs.drop(group_rows.index.difference([largest_timestamp_row.name]))
          self.merge_tfs(group, largest_timestamp_row.memory_id)
          
    def create_dist_matrix(self):
      tfidf_df = self.mc_df.copy(deep=False)
      for token in tfidf_df.columns:
        doc_freq = self.termDictionary.get_doc_freq(token)
        tfidf_df[token] = tfidf_df[token] / doc_freq
      tfidfm = csr_matrix(tfidf_df.values)
      cosine_sim_matrix = cosine_similarity(tfidfm)
      return 1 - cosine_sim_matrix
                
                
    def merge_tfs(self, indices, index_to_merge):
      rows_to_merge = self.mc_df.loc[indices]
      sum_of_rows = rows_to_merge.sum()
      self.mc_df.loc[index_to_merge] = sum_of_rows
      self.mc_df = self.mc_df.drop(indices)


    '''This function removes irrelavant clusters after every 200 observations'''  
    def _cleanup(self):
      expired_mcs = self.mcs[self.mcs['weight'] < 0.7]
      drop_indices = expired_mcs.index
      expired_ids = expired_mcs.memory_id
      self.mcs.drop(index=drop_indices, inplace=True)
      self.mc_df.drop(index=expired_ids, inplace=True)
      terms_to_drop = [term for term, term_obj in self.termDictionary.items() if term_obj.weight.weight < 0.7]
      self.mc_df.drop(columns=terms_to_drop, inplace=True)
      for term in terms_to_drop:
        self.termDictionary.remove_term(term)
      
      
    def merge_clusters_cleanup(self):
      #merging for cleanup process
      if(self.n_merge > 1):
        tr = self.d_merge / self.n_merge
        row_ids = self.mc_df.index.tolist()
        dist_m = self.create_dist_matrix()
        clusters = self.group_similar_clusters(dist_m, row_ids, tr)
        self.merge_similar_clusters(clusters)
        self.d_merge = 0
        self.n_merge = 0
      
      
    