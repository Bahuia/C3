from .evaluation import *

def evaluate_single_example(gold_sql, gold_db, predict):
    dirpath = os.path.dirname(__file__)
    spider_dir = os.path.join(os.path.split(os.path.split(os.path.split(dirpath)[0])[0])[0], 'datasets/spider/')
    db_dir = f'{spider_dir}database/'
    table = f'{spider_dir}tables.json'
    kmaps = build_foreign_key_map_from_json(table)

    evaluator = Evaluator()

    p_str = predict
    g_str, db = gold_sql, gold_db
    db_name = db
    db = os.path.join(db_dir, db, db + ".sqlite")
    schema = Schema(get_schema(db))
    try:
        g_sql = get_sql(schema, g_str)
    except:
        return False

    try:
        p_sql = get_sql(schema, p_str)
    except:
        # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
        p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
        }


    # rebuild sql for value evaluation
    kmap = kmaps[db_name]
    g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)


    exact_score = evaluator.eval_exact_match(p_sql, g_sql)

    if exact_score == 1:
        return True
    elif exact_score == 0:
        return False
    else:
        raise Exception





