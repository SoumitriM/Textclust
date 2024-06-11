from peewee import *
from . import DATABASE
from datetime import datetime
import numpy as np

class BaseModel(Model):
    class Meta:
        database = DATABASE

class MicroCluster(BaseModel):
    id = IntegerField(primary_key=True)
    n_observations = IntegerField(default=1.)
    desc = CharField(default='')
    merged_with = ForeignKeyField('self', backref='merged_ids', lazy_load=False, null=True)
        
    def compute_weight(self):
        terms = [tf.weight for tf in self.tfs_.select(TermFrequency.weight)]
        return np.mean(terms) if len(terms) >= 1 else 0.
        
    def soft_delete(self, other=None):
        # TODO: Write nice last history!
        self.merged_with = other if other is not None else self
        self.save()
    
    def hard_delete(self):
        with DATABASE.atomic():
            TermFrequency.delete().where(TermFrequency.mc == self).execute()
            self.delete_instance()
    
    def __str__(self):
        weight = self.get_weight()
        tfs = self.tfs_.select().order_by(TermFrequency.weight.desc()).execute()
        count = min(len(tfs), 5)
        st = f'Cluster {self.id} has Weight: {weight:.3f} ({count}/{len(tfs)})'
        for tf in tfs[:5]:
            term = Term.select().where(Term.id == tf.term).limit(1).execute()[0]
            st += f'\n - Term \"{term.desc}\" has weight {tf.weight:.2f}.'
        return st

class Term(BaseModel):
    id = IntegerField(primary_key=True)
    desc = CharField(index=True, unique=True)
    
    def __str__(self):
        return f'{desc} ({id})'
    
    def find(desc):
        assert isinstance(desc, str), 'Term.desc must be string!'
        term = Term.select().where(Term.desc == desc).limit(1)
        if len(term) >= 1:
            return term[0]
        else:
            return Term.create(desc=desc)
        
    def find_many(descs):
        assert isinstance(descs, list) and isinstance(descs[0], str), 'Term.desc must be a list of strings!'
        query = {t.desc: t for t in Term.select().where(Term.desc << descs).execute()}
        term_insert = []
        
        for desc in descs:
            if desc not in query:
                new_term = Term(desc=desc)
                term_insert.append(new_term)
        
        with DATABASE.atomic():
            Term.bulk_create(term_insert)
        return Term.select().where(Term.desc << descs).execute()

    
class TermFrequency(BaseModel):
    id = IntegerField(primary_key=True)
    mc = ForeignKeyField(MicroCluster, backref='tfs_', lazy_load=False)
    term = ForeignKeyField(Term, backref='tfs_', lazy_load=False)
    timestep = IntegerField()
    weight = FloatField()
    current = BooleanField(default=True)
    
    def create_local(microcluster, term, timestep, weight):
        term_ = Term.find(term)
        return TermFrequency(mc=microcluster, term=term_, timestep=timestep, weight=weight)
    
    def clone(self, mc=None, term=None, timestep=None, weight=None):
        mc = self.mc if mc is None else mc
        term = self.term if term is None else term
        timestep = self.timestep if timestep is None else timestep
        weight = self.weight if weight is None else weight
        return TermFrequency(mc=mc, mc_current=mc, term=term, timestep=timestep, weight=weight)
    
    def __and__(self, other):
        assert isinstance(other, TermFrequency), f'Other must be of type TermFrequency but is {type(other)}!'
        assert other.timestep == self.timestep, f'Other must be of same timestep but are of {self.timestep}\
                                                  and {other.timestep}!'
        self.weight += other.weight
        return self
    
    def __str__(self):
        desc = self.term.get()[0].desc
        return f'{desc}: {self.weight} at Time {self.timestep} (Cluster {self.mc_id})'

class Weight(BaseModel):
    id = IntegerField(primary_key=True)
    mc = ForeignKeyField(MicroCluster, backref='weights_', lazy_load=False)
    realtime = DateTimeField(default=datetime.now())
    timestep = IntegerField()
    weight = FloatField()
    desc = TextField(default='')
    
    def __str__(self):
        return f'{self.weight} at Time {self.timestep} (Cluster {self.mc_id})\n{self.desc}'
    
def drop_tables(tables=[MicroCluster, Term, TermFrequency]):
    DATABASE.drop_tables(tables)

def create_tables(tables=[MicroCluster, Term, TermFrequency]):
    DATABASE.create_tables(tables)
    
__all__ = ['MicroCluster', 'Term', 'TermFrequency', 'drop_tables', 'create_tables']