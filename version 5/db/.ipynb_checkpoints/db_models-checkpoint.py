from peewee import *
from . import DATABASE

class BaseModel(Model):
    class Meta:
        database = DATABASE

class MicroCluster(BaseModel):
    id = IntegerField(primary_key=True)
    n_observations = IntegerField(default=1)
    weight = FloatField(default=None, null=True)
    old_weight = FloatField(default=None, null=True)
    merged_with = ForeignKeyField('self', backref='merged_ids', lazy_load=False, null=True)

class Term(BaseModel):
    id = IntegerField(primary_key=True)
    desc = TextField()

class TermFrequency(BaseModel):
    id = IntegerField(primary_key=True)
    old = ForeignKeyField('self', lazy_load=False, null=True)
    time = IntegerField()
    weight = FloatField(default=1.)

class MCTermMapper(BaseModel):
    mc = ForeignKeyField(MicroCluster, backref='terms_', lazy_load=True)
    term = ForeignKeyField(Term, backref='mcs_', lazy_load=True)
    tf = ForeignKeyField(TermFrequency, backref='tfs_', lazy_load=True)