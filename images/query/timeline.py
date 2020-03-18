import json
import os
from .utils import *

TIMELINE_SPAN = 10  # If they want more, submit more

def get_timeline(images, timeline_type="after"):
    images = [grouped_info_dict[image]for image in images]
    group_ids = [int(image["group"].split('_')[-1]) for image in images]
    print(timeline_type)
    scenes = []
    if timeline_type == "current":
        min_group = min(group_ids)
        max_group = max(group_ids)
        group_range = range(min_group, max_group + 1)

        date = images[0]["group"].split("_")[0]
        for index in group_range:
            new_group_id = f"{date}_{index}"
            for scene in group_info[date][new_group_id]:
                scenes.append(group_info[date][new_group_id][scene])
    else:
        if timeline_type == "after":
            max_group = max(group_ids)
            group_range = range(max_group + 1, max_group + TIMELINE_SPAN + 1)
        elif timeline_type == "before":
            min_group = min(group_ids)
            group_range = range(max(min_group - TIMELINE_SPAN, 0), min_group)
        else:
            raise NotImplementedError

        date = images[0]["group"].split("_")[0]
        for index in group_range:
            new_group_id = f"{date}_{index}"
            if new_group_id in group_info[date]:
                scenes.append([img for scene in group_info[date][new_group_id]
                                for img in group_info[date][new_group_id][scene]])

    return scenes
