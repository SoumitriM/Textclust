from peewee import SqliteDatabase
from db_models import MicroCluster, Term, TermFrequency, MCTermMapper

DATABASE = SqliteDatabase('file:topictrends.db')

def drop_tables(tables=):
    