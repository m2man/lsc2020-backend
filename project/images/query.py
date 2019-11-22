import images.MyLibrary as mylib

def ESearch(es, List_synonym, input_query):
    query_request_txt, query_request_json = mylib.generate_es_query_dismax_querystringquery(q=input_query,
                                                                                            list_synonym=List_synonym,
                                                                                            max_change=1,
                                                                                            tie_breaker=0.7)

    request_result, id_result = mylib.search_es(es, index="lsc2019", request=query_request_json,
                                                                percent_thres=0.5, max_len=100)

    return id_result