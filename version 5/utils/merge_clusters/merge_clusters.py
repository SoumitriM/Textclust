class MergeClusters:
  def __init__(self, mc, timestep):
    self.mc = mc
    self.timestep = timestep
    
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