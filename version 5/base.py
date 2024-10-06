from peewee import fn
import numpy as np
from datetime import datetime
from memory.models2 import Weight, TermDictionary, MicroCluster, Term, TermFrequency
from sklearn.metrics.pairwise import cosine_similarity
from union_find import find_clusters
import uuid

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
        self.conn = conn
        self.d_merge = 0
        self.n_merge = 0
        self.expired_terms = TermDictionary()
    
    def create_new(self, n_grams, timestep):
        return self.create_new_mc(n_grams, timestep)
    
    def microclusters(self, ignore_ids=[]):
        return self.microClusters(ignore_ids)
    
        
    def _learn_one(self, n_grams, tweet_id, timestep=None, realtime=None):
        ## Save starting time and compute timestep if missing
        if self.use_realtime:
            realtime = datetime.now() if realtime is None else realtime
            self.start_realtime = realtime if self.start_realtime is None else self.start_realtime
            timestep = max(realtime - self.start_realtime, 0)
        else:
            timestep = self.n_observations if timestep is None else timestep
        ## Increase counter and execute step
        self.n_observations += 1
        return self.step(n_grams,tweet_id, timestep)
        
    def step(self, n_grams, tweet_id, timestep):
        ## If n_grams is empty, don't make prediction
        if len(n_grams) == 0:
            return None
        ## Create new MicroCluster
        if timestep%1 == 0:
          self.update_database()
        if timestep%200 == 0:
          if len(self.microClusters) > 2:
            self._cleanup(timestep)
            #self.update_database()
        self.create_new_mc(n_grams, tweet_id, timestep)
        #return 1 # Provide new Cluster ID as prediction
    
    
    def create_new_mc(self, n_grams, tweet_id, timestep):
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
            term_flags[token] = 1
            tf = freq
            term_frequencies[token] = TermFrequency(term_ref=term.memory_id, tf=tf, weight=1, time_stamp=timestep)
        weight_mc = Weight(weight=1)
        tweet_id = [tweet_id]
        mc_new = MicroCluster(time_stamp=timestep, weight=weight_mc, term_frequencies=term_frequencies,tweet_ids=tweet_id, term_flags=term_flags, active=True)
        if(len(self.microClusters) < 2):
          self.microClusters[mc_new.memory_id] = mc_new
        else:
          self.merge_if_eligible(mc_new, timestep)
          
    
    '''This function calculates the threshold value for minimum distance of merging'''
    def calculate_threshold(self, similarityMatrix, maxSimilarityIndex):
      tr = 0
      d_matrix = np.array(1 - similarityMatrix)
      filtered_values = np.delete(d_matrix, maxSimilarityIndex)
      mean_excluding_min = np.mean(filtered_values)
      std_excluding_min = np.std(filtered_values)
      tr = mean_excluding_min - 0.5 * std_excluding_min
      return tr
            
    '''This function calculates cosine similarity and returns dist and the id of the closest mc'''
    def calculate_similarity(self, tf_new, tf_matrix, ids):
        tfidf_new = np.array(tf_new).reshape(1, -1)  # Ensure it's a 2D array (1 row)
        tfidf_matrix = np.array(tf_matrix)  # Multiple rows matrix
        rows, columns = tfidf_matrix.shape
        #print(f"Shape of tfidf_matrix: {rows} rows, {columns} columns")
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_new).flatten()
        min_distance_index = np.argmax(cosine_similarities)
        threshold = self.calculate_threshold(cosine_similarities, min_distance_index)
        return (1 - cosine_similarities[min_distance_index], ids[min_distance_index], threshold)
      
    '''This function merges mc_new to its closest mc which becomes its parent mc'''
    def mergeClusters(self, mc_new, mc_parent):
      new_t_ids = mc_new.tweet_ids
      mc_parent.n_observations += 1
      mc_parent.weight.weight += mc_new.weight.weight
      mc_parent.time_stamp = mc_new.time_stamp
      mc_parent.tweet_ids += new_t_ids
      for key , val in mc_new.term_flags.items():
        mc_parent.term_flags[key] = mc_parent.term_flags[key] | val
      for token, tf_new in mc_new.term_frequencies.items():
        if token in mc_parent.term_frequencies:
          tf_parent = mc_parent.term_frequencies[token]
          tf_parent.tf += tf_new.tf
          tf_parent.weight.weight += 1
          tf_parent.time_stamp = max(tf_new.time_stamp, tf_parent.time_stamp)
        else:
          mc_parent.term_frequencies[token] = tf_new
      mc_new.merged_with = mc_parent.memory_id
      mc_new.active = False
    
    def fade_terms(self, curr_tokens, timestep):
      for token, term_obj in self.termDictionary.items():
        if curr_tokens[token] == 0:
          new_wt = max(0,term_obj.weight.weight * (2 **(-self.fading_factor * (timestep - term_obj.time_stamp))))
          self.termDictionary[token].weight.weight  = new_wt
          
    def fade_tfs(self, curr_tokens, mc_id, timestep):
      mc = self.microClusters[mc_id]
      tfs = mc.term_frequencies
      for token, term_obj in tfs.items():
        if curr_tokens[token] == 0:
          new_wt = max(0,term_obj.weight.weight * (2 **(-self.fading_factor * (timestep - term_obj.time_stamp))))
          tfs[token].weight.weight  = new_wt
          
    '''This function fades existing mcs , calculates tf_idf of the new mc, and the existing mcs and calculates the cosine_similarity.
        If dist< threshold, merge_clusters() is called. Else, mc_new is appended  to the list of mcs.
    '''
    def merge_if_eligible(self, mc_new, timestep):
      tfidfmatrix = []
      ids=[]
      termFlagsNew = mc_new.term_flags
      #fade terms globally
      self.fade_terms(termFlagsNew, timestep)
      tfNew = mc_new.term_frequencies
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
          self.fade_tfs(termFlagsNew, id, timestep)
          tf_idf = [int(mc_tfs[term].tf) / self.termDictionary.get_doc_freq(term) if termFlags[term]==1 else 0 
                  for term, termObj in self.termDictionary.items()]
          ## append 0s to make tf_idf length the same as the length of termDictionary
          #print(len(tf_idf))
          tfidfmatrix.append(tf_idf)
          ids.append(id)

      minDist, minDistIndex, tr = self.calculate_similarity(tfidf_new,tfidfmatrix, ids)
      mc_closest = self.microClusters[minDistIndex]
      if minDist < tr:
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
                                

    def remove_terms(self):
        # Filter the expired term keys
        self.expired_terms = {
          term: term_obj
          for term, term_obj in self.termDictionary.items()
          if term_obj.weight.weight < (2. ** (-0.9))
        }
        # Delete expired_terms key-value pairs from self.termDictionary
        for term in self.expired_terms:
            del self.termDictionary[term]
        for mc in self.microClusters.values():
          # Remove terms from both mc.tfs and mc.term_flags in one step
          for term in self.expired_terms:
            if term in mc.term_frequencies:
              del mc.term_frequencies[term]
            if term in mc.term_flags:
              del mc.term_flags[term]

    def get_distance_matrix(self, tfidf_matrix):
      tfidm = np.array(tfidf_matrix)
      if tfidf_matrix is not None and len(tfidf_matrix) > 0:
        similarity_matrix = cosine_similarity(tfidm)
        return 1 - similarity_matrix
      return []

      
    def create_tfidm(self):
      tfidfmatrix = []
      ids=[]
      for id, mc in self.microClusters.items():
          termFlags = mc.term_flags
          mc_tfs = mc.term_frequencies
          tf_idf = [int(mc_tfs[term].tf) / self.termDictionary.get_doc_freq(term) if termFlags[term]==1 else 0 
                  for term, termObj in self.termDictionary.items()]
          ## add 0s for rest of term.dict.length
          tfidfmatrix.append(tf_idf)
          ids.append(id)
      return (tfidfmatrix,ids)
     
    def merge_grouped_mcs(self, groups):
      for group in groups:
        if len(group) > 1:
          min_timestep_id = min(group, key=lambda id: self.microClusters[id].time_stamp)
          for id in group:
            if id != min_timestep_id:
              self.mergeClusters(self.microClusters[id], self.microClusters[min_timestep_id])
              self.mcs_inactive[id] = self.microClusters[id]
              del self.microClusters[id]
          print("Number of inactive microclusters: ", len(self.mcs_inactive))
    
    def merge_similar_clusters(self, timestep):
      tfidf_m , ids = self.create_tfidm()
      dist_m = self.get_distance_matrix(tfidf_m)
      dist_tr = self.d_merge / self.n_merge
      grouped_clusters_list = find_clusters(dist_m, ids, dist_tr)
      self.merge_grouped_mcs(grouped_clusters_list)

    def remove_tfs(self, mc):
      tfs = mc.term_frequencies
      updated_tfs = {}
      for token, term_obj in tfs.items():
        if term_obj.weight.weight >= (2. ** (-self.fading_factor * self.t_gap)):
          updated_tfs[token] = term_obj
        else:
          mc.term_flags[token] = 0
      mc.term_frequencies = updated_tfs
       
    def _cleanup(self, timestep):
      updated_mcs = {}
      print("Number of microclusters before cleanup: ", len(self.microClusters))
      self.merge_similar_clusters(timestep)
      for mc_id, mc in self.microClusters.items():
        if mc.weight.weight >= (2. ** (-self.fading_factor * self.t_gap)):
          updated_mcs[mc_id] = mc
        else:
          mc.active = False
          self.mcs_inactive[mc_id] = mc
      self.microClusters = updated_mcs    
      print("Number of microclusters after cleanup: ", len(self.microClusters))
      self.remove_terms()
      print("Total active terms: ", len(self.termDictionary))

    
    # def update_terms(self, cur):
    #   update_terms = """
    #   INSERT INTO term (term_id, term, doc_frequency, timestep, weight) 
    #   VALUES (%s, %s, %s, %s)
    #   ON CONFLICT (term_id) DO UPDATE
    #   SET weight = EXCLUDED.weight, timestep = EXCLUDED.timestep, doc_frequency = EXCLUDED.doc_frequency;
    #   """
    #   insert_weight = ''''''
      
    #   # Filter active microclusters and insert them into the table
    #   for term, term_obj in self.termDictionary.items():
    #     cur.execute(update_terms, (term_obj.memory_id, term_obj.weight.weight, term_obj.time_stamp, term_obj.document_frequency))

    #   # Commit the transaction
      
    #   self.conn.commit()
    #   return 0
    
    def update_database(self):
      cur = self.conn.cursor()
      # self.update_term_table(cur)
      # self.update_mc_weight_table(cur)
      
      update_mc_query = """
      INSERT INTO microcluster (mc_id,  timestep, parent_id, tweets, active) 
      VALUES (%s, %s, %s, %s, %s)
      ON CONFLICT (mc_id) DO UPDATE
      SET timestep = EXCLUDED.timestep, parent_id = EXCLUDED.parent_id, tweets = EXCLUDED.tweets, active = EXCLUDED.active;
      """
      update_tf_query = """
      INSERT INTO microcluster (mc_id, weight, timestep, active) 
      VALUES (%s, %s, %s, %s)
      ON CONFLICT (mc_id) DO UPDATE
      SET timestep = EXCLUDED.timestep, active = EXCLUDED.active;
      """
      
      insert_weight_query = """
      INSERT INTO weight_mc (weight_id, weight, timestep, mc_id) 
      VALUES (%s, %s, %s, %s);
      """
      update_terms_query = """INSERT INTO term (term, weight, doc_frequency, timestep)
      VALUES (%s, %s, %s, %s)
      ON CONFLICT (term) DO UPDATE
      SET doc_frequency = EXCLUDED.doc_frequency, timestep = EXCLUDED.timestep, weight = EXCLUDED.weight;
      """
      update_tf_query = """INSERT INTO term_frequencies (mc_id, term, weight, frequency, timestep)
      VALUES (%s, %s, %s, %s, %s)
      """
      
      delete_expired_terms_query = """DELETE FROM term where term = %s"""
      
      
      
      for t, term in self.termDictionary.items():
        cur.execute(update_terms_query, (term.token, term.weight.weight, term.document_frequency, term.time_stamp))

      # Insert active mcs into the table
      for mc in self.microClusters.values():
        #update the tfs
        #update the weights
        weight = mc.weight
        termfreq = mc.term_frequencies
        cur.execute(update_mc_query, (mc.memory_id, mc.time_stamp, mc.merged_with, mc.tweet_ids, True))
        cur.execute(insert_weight_query, (uuid.uuid4(), weight.weight, mc.time_stamp, mc.memory_id))
        for term, tf in termfreq.items():
          cur.execute(update_tf_query, (mc.memory_id, term, tf.weight.weight, tf.tf, tf.time_stamp))
        
      # Insert expired mcs into the table
      for mc in self.mcs_inactive.values():
          #update the tfs
        weight = mc.weight
        cur.execute(update_mc_query, (mc.memory_id, mc.time_stamp, mc.merged_with, mc.tweet_ids, False))
        cur.execute(insert_weight_query, (uuid.uuid4(), weight.weight, mc.time_stamp, mc.memory_id))
        for term, tf in termfreq.items():
          cur.execute(update_tf_query,(mc.memory_id, term, tf.weight.weight, tf.tf, tf.time_stamp))
          
      for t, term in self.expired_terms.items():
        cur.execute(delete_expired_terms_query, (t,))
      self.conn.commit()
      
      return 0

      
    