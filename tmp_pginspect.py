import inspect
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    import langgraph.checkpoint.postgres as pg
    print('PostgresSaver init:', inspect.signature(PostgresSaver.__init__))
    print('has from_conn_string:', hasattr(PostgresSaver, 'from_conn_string'))
    if hasattr(PostgresSaver, 'from_conn_string'):
        print('from_conn_string sig:', inspect.signature(PostgresSaver.from_conn_string))
    print('pg attrs:', [a for a in dir(pg) if 'Postgres' in a])
except Exception as e:
    print('checkpoint err', e)
try:
    from langgraph.store.postgres import PostgresStore
    import langgraph.store.postgres as sp
    print('PostgresStore init:', inspect.signature(PostgresStore.__init__))
    print('sp attrs:', [a for a in dir(sp) if 'Postgres' in a])
except Exception as e:
    print('store err', e)
