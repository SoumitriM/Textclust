from datetime import datetime
import numpy as np
import uuid
from dataclasses import dataclass


class MicroCluster:
    def __init__(self, time_stamp, database_id=None, desc='', weight=1, term_frequencies=None):
        if term_frequencies is None:
            self.term_frequencies = []
        self.database_id = database_id
        self.memory_id = uuid.uuid4().hex
        self.n_observations = 1
        self.desc = desc
        self.merged_with = []
        self.weight = weight
        self.time_stamp = time_stamp
        self.term_frequencies = term_frequencies

    def compute_weight(self):
        pass
        # TODO:?
        # terms = [tf.weight for tf in self.tfs_.select(TermFrequency.weight)]
        # return np.mean(terms) if len(terms) >= 1 else 0.

    def soft_delete(self, other=None):
        pass
        # TODO: Write nice last history!
        # self.merged_with = other if other is not None else self
        # self.save()

    def hard_delete(self):
        pass
        # with DATABASE.atomic():
        #     TermFrequency.delete().where(TermFrequency.mc == self).execute()
        #     self.delete_instance()

    # def __str__(self):
    #     pass
        # weight = self.get_weight()
        # tfs = self.tfs_.select().order_by(TermFrequency.weight.desc()).execute()
        # count = min(len(tfs), 5)
        # st = f'Cluster {self.id} has Weight: {weight:.3f} ({count}/{len(tfs)})'
        # for tf in tfs[:5]:
        #     term = Term.select().where(Term.id == tf.term).limit(1).execute()[0]
        #     st += f'\n - Term \"{term.desc}\" has weight {tf.weight:.2f}.'
        # return st


@dataclass
class Term:
    def __init__(self, desc, document_frequency, database_id=None):
        self.database_id = database_id
        self.memory_id = uuid.uuid4().hex
        self.desc = desc
        self.document_frequency = document_frequency


# terms_dict = {}
# sample_tokens = ['this', 'is', 'a', 'test', 'test']
# for sample_token in sample_tokens:
#     if sample_token not in terms_dict:
#         term = Term(sample_token, 1)
#         terms_dict[sample_token] = term
#     else:
#         terms_dict[sample_token].document_frequency += 1
# print(1)
class TermFrequency:
    def __init__(self, term_memory_id, time_stamp, weight, database_id=None):
        self.database_id = database_id
        self.memory_id = uuid.uuid4().hex
        self.term = term_memory_id
        self.time_stamp = time_stamp
        self.weight = weight

    # def create_local(microcluster, term, timestep, weight):
    #     term_ = Term.find(term)
    #     return TermFrequency(mc=microcluster, term=term_, timestep=timestep, weight=weight)
    #
    # def clone(self, mc=None, term=None, timestep=None, weight=None):
    #     mc = self.mc if mc is None else mc
    #     term = self.term if term is None else term
    #     timestep = self.timestep if timestep is None else timestep
    #     weight = self.weight if weight is None else weight
    #     return TermFrequency(mc=mc, mc_current=mc, term=term, timestep=timestep, weight=weight)
    #
    # def __and__(self, other):
    #     assert isinstance(other, TermFrequency), f'Other must be of type TermFrequency but is {type(other)}!'
    #     assert other.timestep == self.timestep, f'Other must be of same timestep but are of {self.timestep}\
    #                                               and {other.timestep}!'
    #     self.weight += other.weight
    #     return self
    #
    # def __str__(self):
    #     desc = self.term.get()[0].desc
    #     return f'{desc}: {self.weight} at Time {self.timestep} (Cluster {self.mc_id})'
#
#

class Weight:
    def __init__(self, memory_id, weight, database_id=None):
        self.database_id = database_id
        self.memory_id = memory_id
        self.weight = weight

    # def __str__(self):
    #     return f'{self.weight} at Time {self.timestep} (Cluster {self.mc_id})\n{self.desc}'


__all__ = ['MicroCluster', 'Term', 'TermFrequency', 'drop_tables', 'create_tables']
