from peewee import SqliteDatabase

DATABASE = SqliteDatabase('file:topictrends.db', pragmas={
    'journal_mode': 'wal',
    'cache_size': -1 * 64000,  # 64MB
    'foreign_keys': 1,
    'ignore_check_constraints': 0,
    'synchronous': 0})
    
__all__ = [DATABASE]