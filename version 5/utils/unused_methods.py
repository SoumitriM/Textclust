'''This function fades tfs after every 50 observations'''
    # def _fade_tfs(self, timestep, mc_tfs):
    #     threshold_weight = 2. ** (-self.fading_factor * self.t_gap)
    #     updated_mc_tfs = {}
    #     for token, tf in mc_tfs.items():
    #         new_wt = self.fade_weight(timestep, tf)
    #         if (new_wt >= threshold_weight):
    #           tf.time_stamp = timestep
    #           updated_mc_tfs[token] = tf
    #         else:
    #           term = self.termDictionary.get_term(token)
    #           term.weight.weight -= tf.tf
    #           if (term.weight.weight <= 0):
    #             self.termDictionary.remove_term(token)      
    #     return updated_mc_tfs
    

          
'''This function fades clusters after every 50 observations'''     
    # def _fade_clusters(self, timestep):
    #   fading_const = 2. ** (-self.fading_factor)
    #   for id, mc in self.microClusters.items():
    #       time_difference = timestep - mc.time_stamp
    #       new_wt = mc.weight.weight * (fading_const ** time_difference)
    #       mc.weight= Weight(new_wt if new_wt > 0 else 0)
    #       mc.time_stamp = timestep
    #       self.fade_terms()
    #   return self.microClusters