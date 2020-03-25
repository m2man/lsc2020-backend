from .query_types import *
from .utils import *
from ..nlp_utils.extract_info import process_query
from ..nlp_utils.synonym import process_string
from datetime import timedelta, datetime


def query_all(includes, index):
    request = create_base_query(100, includes, {"match_all": {}})
    return post_request(json.dumps(request), index)


def es(query, gps_bounds):
    print(query, gps_bounds)
    if query["before"] and query["after"]:
        last_results = es_three_events(query["current"], query["before"], query["beforewhen"], query["after"], query["afterwhen"], gps_bounds)
    elif query["before"]:
        last_results = es_two_events(query["current"], query["before"], "before", query["beforewhen"], gps_bounds)
    elif query["after"]:
        last_results = es_two_events(query["current"], query["after"], "after", query["afterwhen"], gps_bounds)
    else:
        last_results =individual_es(query["current"], gps_bounds, group_factor="scene")
    return last_results


def individual_es(query, gps_bounds=None, size=1000, extra_filter_scripts=None, group_factor="group"):
    loc, keywords, description, weekday, months, timeofday, activity, region, must_not_terms = process_query(
        query)
    must_terms, should_terms = process_string(description, must_not_terms)
    must_terms.extend(keywords)
    must_terms = list(set(must_terms))
    # Test
    # If not working delete these 2 lines
    should_terms.extend(must_terms)
    should_terms = list(set(should_terms))
    must_terms, should_terms = should_terms, must_terms
    # ===================================
    must_queries = []
    should_queries = []
    filter_queries = []
    must_not_queries = []

    # MUST
    if must_terms:
        must_queries.append({"terms_set": {
                                "descriptions": {
                                    "terms": must_terms,
                                    "minimum_should_match_script": {
                                        "source": "1"
                                    }
                                }
                            }})

    if region:
        must_queries.append({"terms_set": {"region": {"terms": region,
                                                 "minimum_should_match_script": {
                                                        "source": "1"}}}})



    if loc:
        must_queries.append({"match": {"location": {"query": ' '.join(loc)}}})

    # SHOULDS
    if should_terms:
        should_queries.append({"terms_set": {
                            "descriptions": {
                                        "terms": should_terms,
                                        "minimum_should_match_script": {
                                            "source": "1"
                                        }
                                        }
                        }})

    if activity:
        for act in activity:
            if act == "walking":
                should_queries.append({"terms": {"activity": activity}})
            else:
                must_queries.append({"terms": {"activity": activity}})

    # FILTERS
    if weekday:
        filter_queries.append({"terms": {"weekday": weekday}})

    script = extra_filter_scripts if extra_filter_scripts else []
    if timeofday:
        time_script = []
        for t in timeofday:
            if 'morning' in t:
                time_script.append(" (doc['time'].value.getHour() <= 10) ")
            elif "noon" in t:
                time_script.append(
                    " (doc['time'].value.getHour() <= 14 && doc['time'].value.getHour() >= 10) ")
            elif 'afternoon' in t:
                time_script.append(
                    " (doc['time'].value.getHour() <= 17 && doc['time'].value.getHour() >= 12) ")
            elif 'night' in t or 'evening' in t:
                time_script.append(" doc['time'].value.getHour() >= 16 ")
            else:
                print(t)
        if time_script:
            script.append(f'({"||".join(time_script)})')
    if months:
        month_script = []
        for m in months:
            month2num = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
                         "august": 8,
                         "september": 9, "october": 10, "november": 11, "december": 12}
            month_script.append(
                f" doc['time'].value.getMonthValue() == {month2num[m]} ")
        if month_script:
            script.append(f'({"||".join(month_script)})')

    if script:
        script = "&&".join(script)
        filter_queries.append({"script": {
            "script": {
                "source": script
            }}})

    if gps_bounds:
        filter_queries.append(get_gps_filter(gps_bounds))

    # MUST NOT
    if must_not_terms:
        must_not_queries.append({
            "terms": {
                "descriptions": must_not_terms}
            })

    # CONSTRUCT JSON
    main_query = {}
    if must_queries:
        main_query["must"] = must_queries[0] if len(must_queries) == 1 else must_queries
    else:
        main_query["must"] = {"match_all": {}}
    if should_queries:
        main_query["should"] = should_queries[0] if len(should_queries) == 1 else should_queries
    if filter_queries:
        main_query["filter"] = filter_queries[0] if len(filter_queries) == 1 else filter_queries
    if must_not_queries:
        main_query["must_not"] = must_not_queries[0] if len(must_not_queries) == 1 else must_not_queries

    json_query = {
        "size": size,
        "_source": {
            "includes": [
                "image_path",
                "descriptions",
                "activity",
                "location",
                "weekday",
                "time",
                "gps",
                "scene",
                "group",
                "before",
                "after"
            ]
        },
        "query": {"bool": main_query}
    }

    print(json.dumps(json_query), "lsc2020")
    return group_results(post_request(json.dumps(json_query), "lsc2020"), group_factor)


def forward_search(query, conditional_query, condition, time_limit, gps_bounds=None):
    main_events = individual_es(
        query, gps_bounds, size=1000, group_factor="scene")
    extra_filter_scripts = []

    for time_group in find_time_span(main_events):
        if condition == "before":
            time = datetime.strftime(
                time_group["begin_time"], "%Y, %m, %d, %H, %M, %S")
            time = ', '.join([str(int(i)) for i in time.split(', ')])
            time = f"ZonedDateTime.of({time}, 0, ZoneId.of('Z'))"
            script = f" 0 < ChronoUnit.HOURS.between(doc['time'].value, {time}) &&  ChronoUnit.HOURS.between(doc['time'].value, {time}) < {float(time_limit) + 2} "
        else:
            time = datetime.strftime(
                time_group["end_time"], "%Y, %m, %d, %H, %M, %S")
            time = ', '.join([str(int(i)) for i in time.split(', ')])
            time = f"ZonedDateTime.of({time}, 0, ZoneId.of('Z'))"
            script = f" 0 < ChronoUnit.HOURS.between({time}, doc['time'].value) &&  ChronoUnit.HOURS.between({time}, doc['time'].value) < {float(time_limit)+ 2} "
        extra_filter_scripts.append(f"({script})")
    extra_filter_scripts = [f''"||".join(extra_filter_scripts)]
    conditional_events = individual_es(conditional_query, size=10000,
                                       extra_filter_scripts=None)

    return main_events, conditional_events, extra_filter_scripts


def add_pairs(main_events, conditional_events, condition, time_limit):
    pair_events = []
    for main_event in main_events:
        for conditional_event in conditional_events:
            if condition == "after" and timedelta() < conditional_event["begin_time"] - main_event["begin_time"] < timedelta(hours=float(time_limit) + 2):
                pair_events.append({"current": main_event["current"],
                                    "before": main_event["before"],
                                    "after": conditional_event["current"],
                                    "begin_time": main_event["begin_time"],
                                    "end_time": main_event["end_time"],
                                    "gps": main_event["gps"][:2] + [conditional_event["gps"][1]]})
            elif condition == "before" and timedelta() < main_event["begin_time"] - conditional_event["begin_time"] < timedelta(hours=float(time_limit) + 2):
                pair_events.append({"current": main_event["current"],
                                    "before": conditional_event["current"],
                                    "after": main_event["after"],
                                    "begin_time": main_event["begin_time"],
                                    "end_time": main_event["end_time"],
                                    "gps": [conditional_event["gps"][1]] + main_event["gps"][1:]})
    return pair_events


def es_two_events(query, conditional_query, condition, time_limit, gps_bounds, return_extra_filter=False):
    if not time_limit:
        time_limit = "1"
    else:
        time_limit = time_limit.strip("h")
    # Forward search
    main_events, conditional_events, extra_filter_scripts = forward_search(query, conditional_query,
                                                                           condition, time_limit, gps_bounds)
    pair_events = add_pairs(
        main_events, conditional_events, condition, time_limit)

    # Backward search
    conditional_events, main_events, _ = forward_search(
        conditional_query, query, "before" if condition == "after" else "after", time_limit)
    pair_events += add_pairs(main_events,
                             conditional_events, condition, time_limit)

    print(len(pair_events))
    if return_extra_filter:
        return pair_events, extra_filter_scripts
    else:
        return pair_events


def es_three_events(query, before, beforewhen, after, afterwhen, gps_bounds):
    if not afterwhen:
        afterwhen = "1"
    else:
        afterwhen = afterwhen.strip('h')
    if not beforewhen:
        beforewhen = "1"
    else:
        beforewhen = afterwhen.strip('h')
    before_pairs, extra_filter_scripts = es_two_events(
        query, before, "before", beforewhen, gps_bounds, return_extra_filter=True)
    after_events = individual_es(after,
                                 size=5000, extra_filter_scripts=extra_filter_scripts)
    print(len(before_pairs), len(after_events))

    pair_events = []
    for before_pair in before_pairs:
        for after_event in after_events:
            if timedelta() < after_event["begin_time"] - before_pair["end_time"] < timedelta(hours=float(afterwhen) + 2):
                pair_events.append({"current": before_pair["current"],
                                    "before": before_pair["before"],
                                    "after": after_event["current"],
                                    "begin_time": before_pair["begin_time"],
                                    "end_time": before_pair["end_time"],
                                    "gps": before_pair["gps"][:2] + [after_event["gps"][1]]})
    return pair_events


if __name__ == "__main__":
    query = "graveyard in norway oslo"
    loc, keywords, description, weekday, months, timeofday, activity, region, must_not_terms = process_query(
        query)
    must_terms, should_terms = process_string(description, must_not_terms)
