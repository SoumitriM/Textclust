from peewee import fn
from abc import ABC
import numpy as np
from datetime import datetime
from scipy import spatial
import pandas as pd
from collections import defaultdict
from memory.models2 import Weight, TermDictionary, MicroCluster, Term, TermFrequency

class BaseModel():
    start_realtime = None
    last_cleanup = None
    n_observations = 0
    
    
    def __init__(self, fading_factor=0.0005, t_gap=100, auto_cleanup=True, use_realtime=False):
        self.fading_factor = fading_factor
        self.t_gap = t_gap
        self.auto_cleanup = auto_cleanup
        self.use_realtime = use_realtime
        self.termDictionary = TermDictionary()
        self.microClusters = {}
        self.tfs = []
        self.mc_df = pd.DataFrame(columns=[])
        
    
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
        if timestep%50 == 0:
          #print("cleanup...", len(microClusters))
          self._cleanup()
        self._create_new_mc(n_grams, timestep)
        return 1 # Provide new Cluster ID as prediction
    
    
    def _create_new_mc(self, n_grams, timestep):
        term_frequencies = {}
        term_flags = {key: 0 for key in self.termDictionary}
        total_terms = len(n_grams)
        for token, freq in n_grams.items():
            # print(token)
            term = self.termDictionary.get_term(token)
            if term is None:
                term = Term(token=token, weight=1)
                #add term as a new column and put it as 0 for all rows
                self.termDictionary.add_term(term=token, term_obj=term)
                for mc, mc_val in self.microClusters.items():
                  mc_val.term_flags[token] = 0
            else:
              term.weight.weight += freq
            term_flags[token] = 1
            #create mc
            #add mc_id as new row
            #put the freq values for each col=term
            tf = freq
            weight_tf = Weight(weight=1)
            term_frequencies[token] = TermFrequency(term_ref=term.memory_id, tf=tf, time_stamp = timestep, weight=weight_tf)
            term.document_frequency += 1
        weight_mc = Weight(weight=1)
        mc_new = MicroCluster(time_stamp=timestep, weight=weight_mc, term_frequencies=term_frequencies, term_flags=term_flags)
        if(len(self.microClusters) < 2):
          self.microClusters[mc_new.memory_id] = mc_new
        else:
          self.create_merge_clusters(mc_new, timestep)

    
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
      tfidf_new = np.array(tf_new)
      tfidfmatrix = np.array(tf_matrix)
      tfidf_new_norm = tfidf_new / np.linalg.norm(tfidf_new)
      tfidfmatrix_norm = tfidfmatrix / np.linalg.norm(tfidfmatrix, axis=1)[:, np.newaxis]
      cosine_similarities = np.dot(tfidfmatrix_norm, tfidf_new_norm)
      min_distance_index = np.argmax(cosine_similarities)
      threshold = self.calculate_threshold(cosine_similarities, min_distance_index)
      return (1 - cosine_similarities[min_distance_index], ids[min_distance_index], threshold)
      
    '''This function merges mc_new to its closest mc which becomes its parent mc'''
    def mergeClusters(self, mc_new, mc_parent):
      mc_parent.n_observations +=1
      mc_parent.merged_with = mc_new.memory_id
      mc_parent.weight.weight +=1
      mc_parent.time_stamp = mc_new.time_stamp
      for key , val in mc_new.term_flags.items():
        mc_parent.term_flags[key] = mc_parent.term_flags[key] | val
      merged_dict = defaultdict(int)
      for token, tf_new in mc_new.term_frequencies.items():
        if token in mc_parent.term_frequencies:
          mc_parent.term_frequencies[token].time_stamp = tf_new.time_stamp
          mc_parent.term_frequencies[token].weight.weight +=1
          mc_parent.term_frequencies[token].tf += tf_new.tf
        else:
          mc_parent.term_frequencies[token] = tf_new  
      return 0
    
    '''This function calculates tf_idf of the new mc and the existing mcs and calculates the cosine_similarity.
        If dist< threshold, merge_clusters() is called. Else, mc_new is appended  to the list of mcs.
    '''
    def create_merge_clusters(self, mc_new, timestep):
      tfidfmatrix = []
      ids=[]
      termFlagsNew = mc_new.term_flags
      tfNew = mc_new.term_frequencies
      tfidf_new = [int(tfNew[token].tf) / self.termDictionary.get_doc_freq(token) if flag else 0 
                  for token, flag in termFlagsNew.items()]
      fading_const = 2. ** (-self.fading_factor)
      for id, mc in self.microClusters.items():
          time_difference = timestep - mc.time_stamp
          new_wt = mc.weight.weight * (fading_const ** time_difference)
          mc.weight= Weight(new_wt if new_wt > 0 else 0)
          mc.time_stamp = timestep
          mc.term_frequencies = self._fade_tfs(timestep, mc.term_frequencies)
          termFlags = mc.term_flags
          mc_tfs = mc.term_frequencies
          tf_idf = [int(mc_tfs[token].tf) / self.termDictionary.get_doc_freq(token) if termFlags.get(token, False) else 0 
                    for token, flag in termFlagsNew.items()]
          tfidfmatrix.append(tf_idf)
          ids.append(id)
      
      minDist, minDistIndex, tr = self.calculate_similarity(tfidf_new,tfidfmatrix, ids)
      mc_closest = self.microClusters[minDistIndex]
      if minDist < tr:
        self.mergeClusters(mc_new, mc_closest )
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
    
    def fade_terms(self, curr_tokens, timestep):
      for token, term_obj in self.termDictionary.items():
        if token not in curr_tokens:
          new_wt = max(0,term_obj.weight.weight * (2 **(-self.fading_factor * (timestep - term_obj.time_stamp))))
          self.termDictionary[token].weight.weight  = new_wt
          
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
                                

    '''This function removes irrelavant clusters after every 200 observations'''  
    def _cleanup(self):
      print("length terms", self.termDictionary.getSize()) 
      updated_mcs = {}
      for mc_id, mc in self.microClusters.items():
        if mc.weight.weight >= (2. ** (-self.fading_factor * self.t_gap)):
          updated_mcs[mc_id] = mc
      print(len(updated_mcs))
      self.microClusters = updated_mcs
      #cleanup tfs
      #print(len(self.microClusters))
      
    