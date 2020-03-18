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
    should_query = {"terms_set": {
        "descriptions": {
                    "terms": should_terms,
                    "minimum_should_match_script": {
                        "source": f"{min(3, len(should_terms) - 1 if should_terms else 0)}"
                    }
                    }
    }}
    must_query = {"terms_set": {
        "descriptions": {
            "terms": must_terms,
            "minimum_should_match_script": {
                "source": f"{min(3, max(1, len(must_terms) - 1))}"
            }
        }
    }}
    if not must_terms and not should_terms:
        main_query = {"bool": {}}
    elif must_terms and should_terms:
        main_query = {"bool": {"must": must_query,
                               "should": should_query
                               }}
    elif must_terms:
        main_query = {"bool": {"must": must_query}}
    else:
        main_query = {"bool": {"should": should_query}}

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
        "query": main_query
    }
    filters = []
    if weekday:
        filters.append({"terms": {"weekday": weekday}})
    if region:
        filters.append({"terms": {"region": region}})

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
        filters.append({"script": {
            "script": {
                "source": script
            }}})

    if gps_bounds:
        filters.append(get_gps_filter(gps_bounds))

    more_should = []
    if activity:
        if "must" in json_query["query"]["bool"]:
            more_should.append({"terms": {"activity": activity}})
        else:
            json_query["query"]["bool"]["must"] = {
                "terms": {"activity": activity}}

    if filters:
        json_query["query"]["bool"]["filter"] = filters

    if loc:
        more_should.append({"match": {"location": {"query": ' '.join(loc)}}})

    if more_should:
        if "should" in json_query["query"]["bool"]:
            json_query["query"]["bool"]["should"] = [
                json_query["query"]["bool"]["should"]] + more_should
        else:
            json_query["query"]["bool"]["should"] = more_should

    if must_not_terms:
        json_query["query"]["bool"]["must_not"] = {
            "terms": {
                "descriptions": must_not_terms
            }}

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
