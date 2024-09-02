from peewee import fn
from abc import ABC
import numpy as np
from datetime import datetime
from memory.models import Weight, TermDictionary, MicroCluster, Term, TermFrequency

class BaseModel(ABC):
    start_realtime = None
    last_cleanup = None
    n_observations = 0
    tfs = []
    
    
    def __init__(self, fading_factor=0.0005, t_gab=100, auto_cleanup=True, use_realtime=False):
        self.fading_factor = fading_factor
        self.t_gab = t_gab
        self.auto_cleanup = auto_cleanup
        self.use_realtime = use_realtime
        self.termDictionary = TermDictionary()
        
    def learn_one(self, *args, **kwargs):
        yield 'Not implemented!'
        pass
    
    def cleanup(self, timestep, ignore_ids=[]):
        yield 'Not implemented!'
        pass
    
    def compute_distances(self, microclusters, other=None):
        yield 'Not implemented!'
        pass
    
    def initial_merge(self, microclusters, new_cluster):
        yield 'Not implemented!'
        pass
    
    def init_database(self):
        yield 'Not implemented!'
        pass
    
    def _supervised(self):
        return False
    
    def create_new(self, n_grams, timestep):
        return self._create_new_mc(n_grams, timestep)
    
    def fade_clusters(self, timestep, ignore_ids=[], greedy=False):
        return self._fade_clusters(timestep, ignore_ids=ignore_ids, greedy=greedy)
    
    def microclusters(self, ignore_ids=[]):
        return self._microclusters(ignore_ids)
           
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
        mc_new = self.createNew(n_grams, timestep)
        
        ## Fade oldest TFs here, greedly (save some computation time)
        m_clusters = self.fadeClusters(timestep, ignore_ids=[mc_new.id], greedy=True)
        if len(m_clusters) == 0:
            return mc_new.id
        
        ## Perform cleanup if necessary
        if self.auto_cleanup and self.last_cleanup is None:
            self.last_cleanup = timestep
        elif self.auto_cleanup and (timestep - self.last_cleanup) >= self.t_gab:
            self.cleanup(timestep, ignore_ids=[mc_new.id])
            
        ## Compute Cosine Distances and get min. distance
        merge_with = self.initial_merge(m_clusters, mc_new)
        if merge_with is not None: # Merge!
            m_clusters[merge_with] |= mc_new
            mc_new.hard_delete() # Hard Delete new Clusters (otherwise we will have too many MCs)
            return m_clusters[merge_with].id # Provide Cluster ID as prediction
        return mc_new.id # Provide new Cluster ID as prediction
    

    # creates a new micro-cluster
    def _create_new_mc(self, n_grams, timestep):
        term_frequencies = {}
        for token, freq in n_grams.items():
            term = self.termDictionary.get_term(token)
            if term is None:
                term = Term(token, 1)
                self.termDictionary.add_term(term)
            weight_tf = Weight(freq)
            term_frequencies[token] = TermFrequency(term.memory_id, weight_tf, 1)
            term.document_frequency += 1
        weight_mc = Weight(1)
        mc_new = MicroCluster(timestep, weight_mc, term_frequencies)
        # for term, frequency in n_grams.items(): 
        #     tf_ = TermFrequency.create_local(mc=mc_new, term=term, timestep=timestep, weight=frequency)
        #     self.tfs.append(tf_)
        return mc_new
    
    def _merge_clusters(self, microcluster, other):
        ## Collect all idx of tfs
        mc_idx = [i for i, tf in enumerate(self.tfs) if tf.mc_id == microcluster.id]
        other_idx = [i for i, tf in enumerate(self.tfs) if tf.mc_id == other.id]
        
        ## Merge elemente-wise
        for idx in other_idx:
            tf_ = [self.tfs[i] for i in other_idx if self.tfs[i].term_id == self.tfs[idx].term_id]
            if len(tf_) >= 1:
                self.tfs[idx] = self.tfs[idx] & tf_[0] # Merge terms
            
    
    def _microclusters(self, ignore_ids=[]):
        return [m for m in MicroCluster.select()\
                                       .where(MicroCluster.merged_with.is_null() & \
                                            ~(MicroCluster.id << ignore_ids)).execute()]
    
    
    def _fade_clusters(self, timestep, ignore_ids=[], synchronize=False):
        ## Fade all locally saved TFs and remove all TFs with low importance
        tfs_ = []
        for tf in self.tfs:
            tf.weight *= 2. ** (-self.fading_factor * (timestep - tf.timestep))
            tf.timestep = timestep
            if tf.weight >= (2. ** (-self.fading_factor * self.t_gab)):
                tfs_.append(tf)
         
        ## If synchronize with DB,
        if synchronize:
            with DATABASE.atomic(): 
                TermFrequency.bulk_insert(tfs_, batch_size=256)
            self.tfs = [tf.clone() for tf in tfs_] # Clone existing ones
        else:
            self.tfs = tfs_
            
        return self.microclusters(ignore_ids=ignore_ids) # Load all MC from DB
                                
    #def _fadeClusters(self, timestep, ignore_ids=[], greedy=False):
    #    tfs = TermFrequency.select()\
    #                       .where(~(TermFrequency.term << ignore_ids) &\
    #                               (TermFrequency.timestep < timestep))
    #    if greedy: # Just take the oldest ones and fade those!
    #        tfs.order_by(TermFrequency.timestep).limit(100) # TODO: make limit variable
    #    tfs = tfs.execute()
    #    
    #    for tf in tfs:
    #        tf.weight *= 2. ** (-self.fading_factor * (timestep - tf.timestep))
    #        tf.timestep = timestep
    #    
    #    with DATABASE.atomic():
    #        TermFrequency.bulk_update(tfs, fields=[TermFrequency.weight, TermFrequency.timestep], batch_size=128)
    #    if not greedy: # If we are nor in greedy mode, we delete all TFs with low threshold!
    #        with DATABASE.atomic():
    #            TermFrequency.delete().where(TermFrequency.weight < (2. ** (-self.fading_factor * self.t_gab))).execute()
    #    return self.microclusters(ignore_ids=ignore_ids)