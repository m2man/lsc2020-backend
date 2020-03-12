from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer
from parsedatetime import Constants

from ..nlp_utils.common import *


class TimeTagger:
    def __init__(self):
        regex_lib = Constants()
        self.all_regexes = []
        for key, r in regex_lib.cre_source.items():
            # if key in ["CRE_MODIFIER"]:
            #     self.all_regexes.append(("TIMEPREP", r))
            if key in ["CRE_TIMEHMS", "CRE_TIMEHMS2",
                       "CRE_RTIMEHMS", "CRE_RTIMEHMS"]:
                self.all_regexes.append(("TIME", r))  # TIME (proper time oclock)
            elif key in ["CRE_DATE", "CRE_DATE3", "CRE_DATE4", "CRE_MONTH", "CRE_DAY", "",
                         "CRE_RDATE", "CRE_RDATE2"]:
                self.all_regexes.append(("DATE", r))  # DATE (day in a month)
            elif key in ["CRE_TIMERNG1", "CRE_TIMERNG2", "CRE_TIMERNG3", "CRE_TIMERNG4",
                         "CRE_DATERNG1", "CRE_DATERNG2", "CRE_DATERNG3", "CRE_NLP_PREFIX"]:
                self.all_regexes.append(("TIMERANGE", r))  # TIMERANGE
            elif key in ["CRE_UNITS", "CRE_QUNITS"]:
                self.all_regexes.append(("PERIOD", r))  # PERIOD
            elif key in ["CRE_UNITS_ONLY"]:
                self.all_regexes.append(("TIMEUNIT", r))  # TIMEUNIT
            elif key in ["CRE_WEEKDAY"]:
                self.all_regexes.append(("WEEKDAY", r))  # WEEKDAY
        # Added by myself
        self.all_regexes.append(("TIMEOFDAY", r"\b(afternoon|noon|morning|evening|night|twilight)\b"))
        self.all_regexes.append(("TIMEPREP", r"\b(before|after|while|late|early)\b"))

    def merge_interval(self, intervals):
        if intervals:
            intervals.sort(key=lambda interval: interval[0])
            merged = [intervals[0]]
            for current in intervals:
                previous = merged[-1]
                if current[0] <= previous[1] and current[-1] == previous[-1]:
                    if current[1] > previous[1]:
                        previous[1] = current[1]
                        previous[2] = current[2]
                else:
                    merged.append(current)
            return merged
        return []

    def find_time(self, sent):
        results = []
        for kind, r in self.all_regexes:
            for t in find_regex(r, sent):
                results.append([*t, kind])
        return self.merge_interval(results)

    def tag(self, sent):
        times = self.find_time(sent)
        intervals = dict([(time[0], time[1]) for time in times])
        tag_dict = dict([(time[2], time[3]) for time in times])
        tokenizer = WordPunctTokenizer()
        # for a in [time[2] for time in times]:
        #     tokenizer.add_mwe(a.split())

        # --- FIXED ---
        original_tokens = tokenizer.tokenize(sent)
        original_tags = pos_tag(original_tokens)
        # print(original_tags)
        # --- END FIXED ---

        tokens = []
        current = 0
        for span in tokenizer.span_tokenize(sent):
            if span[0] < current:
                continue
            if span[0] in intervals:
                tokens.append(f'__{sent[span[0]: intervals[span[0]]]}')
                current = intervals[span[0]]
            else:
                tokens.append(sent[span[0]:span[1]])
                current = span[1]

        tags = pos_tag(tokens)

        new_tags = []
        for word, tag in tags:
            if word[:2] == '__':
                new_tags.append((word[2:], tag_dict[word[2:]]))
            else:
                tag = [t[1] for t in original_tags if t[0] == word][0]  # FIXED
                new_tags.append((word, tag))
        return new_tags
