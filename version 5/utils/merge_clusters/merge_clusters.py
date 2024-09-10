class MergeClusters:
  def __init__(self, mc_new, mc_parent, timestep):
    self.mc_new = mc_new
    self.mc_parent = mc_parent
    self.timestep = timestep
    
    '''This function merges mc_new to its closest mc which becomes its parent mc'''
  def mergeClusters(self):
    new_t_ids = self.mc_new.tweet_ids
    self.mc_parent.n_observations += 1
    self.mc_parent.weight.weight += self.mc_new.weight.weight
    self.mc_parent.time_stamp = self.mc_new.time_stamp
    self.mc_parent.tweet_ids += new_t_ids
    for key , val in self.mc_new.term_flags.items():
      self.mc_parent.term_flags[key] = self.mc_parent.term_flags[key] | val
    for token, tf_new in self.mc_new.term_frequencies.items():
      if token in self.mc_parent.term_frequencies:
        tf_parent = self.mc_parent.term_frequencies[token]
        tf_parent.tf += tf_new.tf
        tf_parent.weight.weight += 1
        tf_parent.time_stamp = max(tf_new.time_stamp, tf_parent.time_stamp)
      else:
        self.mc_parent.term_frequencies[token] = tf_new
    self.mc_new.merged_with = self.mc_parent.memory_id
    self.mc_new.active = False
    
    def fade_terms(self, curr_tokens, timestep):
      for token, term_obj in self.termDictionary.items():
        if curr_tokens[token] == 0:
          new_wt = max(0,term_obj.weight.weight * (2 **(-self.fading_factor * (timestep - term_obj.time_stamp))))
          self.termDictionary[token].weight.weight  = new_wt
          
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
          mc.weight.weight = new_wt if new_wt > 0 else 0
          mc.time_stamp = timestep
          termFlags = mc.term_flags
          mc_tfs = mc.term_frequencies
          self.fade_tfs(termFlagsNew, id, timestep)
          tf_idf = [int(mc_tfs[token].tf) / self.termDictionary.get_doc_freq(token) if termFlags[token]==1 else 0 
                  for token, flag in termFlags.items()]
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