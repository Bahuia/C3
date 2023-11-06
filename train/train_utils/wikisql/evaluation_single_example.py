from .evaluation import *

def evaluate_single_example(gold_sql, gold_db, predict):
    dirpath = os.path.dirname(__file__)
    wikisql_dir = os.path.join(os.path.split(os.path.split(os.path.split(dirpath)[0])[0])[0], 'datasets/wikisql/')
    table = f'{wikisql_dir}tables.json'
    kmaps = build_foreign_key_map_from_json(table)

    tables = json.load(open(table, "r"))
    tables = {x["db_id"]: x for x in tables}


    # plist = [("select max(Share),min(Share) from performance where Type != 'terminal'", "orchestra")]
    # glist = [("SELECT max(SHARE) ,  min(SHARE) FROM performance WHERE TYPE != 'Live final'", "orchestra")]
    evaluator = Evaluator()

    p_str = predict
    g_str, db = gold_sql, gold_db
    db_name = db

    p_str = wrap_pred_query(p_str, db_name)
    g_str = wrap_pred_query(g_str, db_name)

    schema = Schema(get_schema(wikisql_dir))

    try:
        g_sql = get_sql(schema, g_str, tables[db])
    except:
        return False


    try:
        p_sql = get_sql(schema, p_str, tables[db])
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

        # print("eval_err_num:{}".format(eval_err_num))

    # rebuild sql for value evaluation
    kmap = kmaps[db_name]

    g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    try:
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
    except:
        return False

    exact_score = evaluator.eval_exact_match(p_sql, g_sql)

    if exact_score == 1:
        return True
    elif exact_score == 0:
        return False
    else:
        raise Exception