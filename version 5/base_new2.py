from peewee import fn
from abc import ABC
import numpy as np
from datetime import datetime
from scipy import spatial
import pandas as pd
from collections import defaultdict
from memory.models2 import Weight, TermDictionary, MicroCluster, Term, TermFrequency
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from union_find import find_clusters

class BaseModel():
  
    start_realtime = None
    last_cleanup = None
    n_observations = 0

    
    
    def __init__(self, conn, fading_factor=0.0005, t_gap=500, auto_cleanup=True, use_realtime=False):
        self.fading_factor = fading_factor
        self.t_gap = t_gap
        self.auto_cleanup = auto_cleanup
        self.use_realtime = use_realtime
        self.termDictionary = TermDictionary()
        self.microClusters = {}
        self.tfs = []
        self.mcs_inactive = {}
        self.mcs_removed = {}
        self.conn = conn
        self.d_merge = 0
        self.n_merge = 0
        
    
    def create_new(self, n_grams, timestep):
        return self._create_new_mc(n_grams, timestep)
    
    def fade_clusters(self, timestep, ignore_ids=[], greedy=False):
        return self._fade_clusters(timestep, ignore_ids=ignore_ids, greedy=greedy)
    
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
        if timestep%200 == 0:
          if len(self.microClusters) > 2:
            self._cleanup(timestep)
          #self.update_database()
        if timestep%7999 == 0:
          print(timestep,"       ",len(self.microClusters))
        self._create_new_mc(n_grams, timestep)
        #return 1 # Provide new Cluster ID as prediction
    
    
    def _create_new_mc(self, n_grams, timestep):
        term_frequencies = {}
        term_flags = {key: 0 for key in self.termDictionary}
        for token, freq in n_grams.items():
            term = self.termDictionary.get_term(token)
            if term is None:
                term = Term(token=token, weight=1, document_frequency=1, time_stamp=timestep)
                #add term as a new column and put it as 0 for all rows
                self.termDictionary.add_term(term=token, term_obj=term)
                for mc, mc_val in self.microClusters.items():
                  mc_val.term_flags[token] = 0
            else:
              self.termDictionary[token].weight.weight += 1
              self.termDictionary[token].document_frequency += 1
              self.termDictionary[token].time_stamp = timestep
              self.termDictionary[token].weight.weight += 1
              
            term_flags[token] = 1
            tf = freq
            term_frequencies[token] = TermFrequency(term_ref=term.memory_id, tf=tf)
        weight_mc = Weight(weight=1)
        mc_new = MicroCluster(time_stamp=timestep, weight=weight_mc, term_frequencies=term_frequencies, term_flags=term_flags, active=True)
        if(len(self.microClusters) < 2):
          self.microClusters[mc_new.memory_id] = mc_new
        else:
          self.merge_if_eligible(mc_new, timestep)
          
    
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
            
    '''This function calculates cosine similarity and returns dist and the id of the closest mc'''
    def calculate_similarity(self, tf_new, tf_matrix, ids):
        # Convert inputs to numpy arrays
        tfidf_new = np.array(tf_new).reshape(1, -1)  # Ensure it's a 2D array (1 row)
        tfidf_matrix = np.array(tf_matrix)  # Multiple rows matrix
        
        # Calculate cosine similarities between tf_new and each row in tf_matrix
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_new).flatten()
        
        # Find the index of the maximum similarity
        min_distance_index = np.argmax(cosine_similarities)
        
        # Calculate a threshold based on the similarities
        threshold = self.calculate_threshold(cosine_similarities, min_distance_index)
        # Return 1 - max similarity, the corresponding id, and the threshold
        return (1 - cosine_similarities[min_distance_index], ids[min_distance_index], threshold)
      
    '''This function merges mc_new to its closest mc which becomes its parent mc'''
    def mergeClusters(self, mc_new, mc_parent):
      mc_parent.n_observations += 1
      mc_parent.weight.weight += mc_new.weight.weight
      mc_parent.time_stamp = mc_new.time_stamp
      for key , val in mc_new.term_flags.items():
        mc_parent.term_flags[key] = mc_parent.term_flags[key] | val
      for token, tf_new in mc_new.term_frequencies.items():
        if token in mc_parent.term_frequencies:
          mc_parent.term_frequencies[token].tf += tf_new.tf
        else:
          mc_parent.term_frequencies[token] = tf_new
      mc_new.merged_with = mc_parent.memory_id
      mc_new.active = False
    
    def fade_terms(self, curr_tokens, timestep):
      for token, term_obj in self.termDictionary.items():
        if curr_tokens[token] == 0:
          new_wt = max(0,term_obj.weight.weight * (2 **(-self.fading_factor * (timestep - term_obj.time_stamp))))
          self.termDictionary[token].weight.weight  = new_wt
          
    '''This function fades existing mcs , calculates tf_idf of the new mc, and the existing mcs and calculates the cosine_similarity.
        If dist< threshold, merge_clusters() is called. Else, mc_new is appended  to the list of mcs.
    '''
    def merge_if_eligible(self, mc_new, timestep):
      tfidfmatrix = []
      ids=[]
      termFlagsNew = mc_new.term_flags
      self.fade_terms(termFlagsNew, timestep)
      tfNew = mc_new.term_frequencies
      # self.fade_terms(termFlagsNew, timestep)
      tfidf_new = [int(tfNew[token].tf) / self.termDictionary.get_doc_freq(token) if flag else 0 
                  for token, flag in termFlagsNew.items()]
      fading_const = 2. ** (-self.fading_factor)
      for id, mc in self.microClusters.items():
          time_difference = timestep - mc.time_stamp
          new_wt = mc.weight.weight * (fading_const ** time_difference)
          mc.weight= Weight(new_wt if new_wt > 0 else 0)
          mc.time_stamp = timestep
          termFlags = mc.term_flags
          mc_tfs = mc.term_frequencies
          tf_idf = [int(mc_tfs[token].tf) / self.termDictionary.get_doc_freq(token) if termFlags[token]==1 else 0 
                  for token, flag in termFlags.items()]
          tfidfmatrix.append(tf_idf)
          ids.append(id)
      minDist, minDistIndex, tr = self.calculate_similarity(tfidf_new,tfidfmatrix, ids)
      mc_closest = self.microClusters[minDistIndex]
      if minDist < tr:
        #self.mcs_inactive[mc_new.memory_id] = mc_new
        self.mergeClusters(mc_new, mc_closest)
        self.n_merge +=1
        self.d_merge += minDist
      else:
        self.microClusters[mc_new.memory_id] = mc_new


    def fade_weight(self, timestep, token):
      fading_const = 2. ** (-self.fading_factor)
      time_difference = timestep - token.time_stamp
      new_wt = token.weight.weight * (fading_const ** time_difference)
      return new_wt
      
    '''This function fades tfs after every 50 observations'''
    def _fade_tfs(self, timestep, mc_tfs):
        threshold_weight = 2. ** (-self.fading_factor * self.t_gap)
        updated_mc_tfs = {}
        for token, tf in mc_tfs.items():
            new_wt = self.fade_weight(timestep, tf)
            if (new_wt >= threshold_weight):
              tf.time_stamp = timestep
              updated_mc_tfs[token] = tf
            else:
              term = self.termDictionary.get_term(token)
              term.weight.weight -= tf.tf
              if (term.weight.weight <= 0):
                self.termDictionary.remove_term(token)      
        return updated_mc_tfs
    

          
    '''This function fades clusters after every 50 observations'''     
    def _fade_clusters(self, timestep, ignore_ids=[], greedy=False):
      fading_const = 2. ** (-self.fading_factor)
      for id, mc in self.microClusters.items():
          time_difference = timestep - mc.time_stamp
          new_wt = mc.weight.weight * (fading_const ** time_difference)
          mc.weight= Weight(new_wt if new_wt > 0 else 0)
          mc.time_stamp = timestep
          self.fade_terms()
          #implement ignore_ids
      return self.microClusters
                                

    def remove_terms(self):
        # Filter the expired term keys
        expired_terms = [term for term, term_obj in self.termDictionary.items() if term_obj.weight.weight < (2. ** (-self.fading_factor * self.t_gap))]
        # Delete expired_terms key-value pairs from self.termDictionary
        for term in expired_terms:
            del self.termDictionary[term]

        # R
        # emove expired terms key-value pairs from mc.tfs and mc.term_flags in one go
        for mc_id, mc in self.microClusters.items():
            # Remove terms from both mc.tfs and mc.term_flags in one step
            for term in expired_terms:
                if term in mc.term_frequencies:
                    del mc.term_frequencies[term]
                if term in mc.term_flags:
                    del mc.term_flags[term]
    
    # def create_dist_matrix(self):
    #   tfidf_df = self.mc_df.copy(deep=False)
    #   for token in tfidf_df.columns:
    #     doc_freq = self.termDictionary.get_doc_freq(token)
    #     tfidf_df[token] = tfidf_df[token] / doc_freq
    #   tfidfm = csr_matrix(tfidf_df.values)
    #   cosine_sim_matrix = cosine_similarity(tfidfm)
    #   return 1 - cosine_sim_matrix
    
    '''This function removes irrelavant clusters after every 200 observations'''  
    
    def get_distance_matrix(self, tfidf_matrix):
    # Calculate dot product similarity matrix
      tfidm = np.array(tfidf_matrix)
      if tfidf_matrix is not None and len(tfidf_matrix) > 0:
        similarity_matrix = cosine_similarity(tfidm)
        #print("here..",similarity_matrix)
        return 1 - similarity_matrix
      return []

      
    def create_tfidm(self, timestep):
      tfidfmatrix = []
      ids=[]
      for id, mc in self.microClusters.items():
          termFlags = mc.term_flags
          mc_tfs = mc.term_frequencies
          tf_idf = [int(mc_tfs[token].tf) / self.termDictionary.get_doc_freq(token) if termFlags[token]==1 else 0 
                  for token, flag in termFlags.items()]
          tfidfmatrix.append(tf_idf)
          ids.append(id)
      #print(tfidfmatrix)
      return (tfidfmatrix,ids)
     
    def merge_grouped_mcs(self, groups):
      for group in groups:
        if len(group) > 1:
          min_timestep_id = min(group, key=lambda id: self.microClusters[id].time_stamp)
          for id in group:
            if id != min_timestep_id:
              self.mergeClusters(self.microClusters[min_timestep_id], self.microClusters[id])
              self.mcs_inactive[id] = self.microClusters[id]
              del self.microClusters[id]
          print("inactive mcs->", len(self.mcs_inactive))
    
    def merge_similar_clusters(self, timestep):
      tfidf_m , ids = self.create_tfidm(timestep)
      dist_m = self.get_distance_matrix(tfidf_m)
      dist_tr = self.d_merge / self.n_merge
      #print(dist_tr)
      grouped_clusters_list = find_clusters(dist_m, ids, dist_tr)
      self.merge_grouped_mcs(grouped_clusters_list)
      #print(grouped_clusters_list)
      #self.merge_clusters(grouped_clusters_list)
      
    def _cleanup(self, timestep):
      updated_mcs = {}
      print("before..", len(self.microClusters))
      self.merge_similar_clusters(timestep)
      for mc_id, mc in self.microClusters.items():
        if mc.weight.weight >= (2. ** (-self.fading_factor * self.t_gap)):
          updated_mcs[mc_id] = mc
        else:
          self.mcs_removed[mc_id] = mc
      self.microClusters = updated_mcs
      print("after..",timestep,"       ",len(self.microClusters))
      # self.create_dist_matrix()
      self.remove_terms()
    
    def update_terms(self, cur):
      update_terms = """
      INSERT INTO term (term_id, term, doc_frequency, timestep, weight) 
      VALUES (%s, %s, %s, %s)
      ON CONFLICT (term_id) DO UPDATE
      SET weight = EXCLUDED.weight, timestep = EXCLUDED.timestep, doc_frequency = EXCLUDED.doc_frequency;
      """
      insert_weight = ''''''
      
      # Filter active microclusters and insert them into the table
      for term, term_obj in self.termDictionary.items():
        cur.execute(update_terms, (term_obj.memory_id, term_obj.weight.weight, term_obj.time_stamp, term_obj.document_frequency))

      # Commit the transaction
      
      self.conn.commit()
      return 0
    
    def update_database(self):
      cur = self.conn.cursor()
      self.update_terms(cur)
      update_mc = """
      INSERT INTO microcluster (mc_id, weight, timestep, active) 
      VALUES (%s, %s, %s, %s)
      ON CONFLICT (mc_id) DO UPDATE
      SET weight = EXCLUDED.weight, timestep = EXCLUDED.timestep, active = EXCLUDED.active;
      """
      insert_weight = '''INSERT INTO weight (mc_id, weight, timestep)'''
      
      # Insert active mcs into the table
      for mem_id, mc in self.microClusters.items():
              #update the term
              cur.execute(update_mc, (mc.memory_id, mc.weight.weight, mc.time_stamp, True))
      # Insert expired mcs into the table
      for mem_id, mc in self.mcs_removed.items():
              cur.execute(update_mc, (mc.memory_id, mc.weight.weight, mc.time_stamp, False))
      # Commit the transaction
      
      self.conn.commit()
      
      return 0

      
    