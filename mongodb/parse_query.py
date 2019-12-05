from datetime import datetime
import parsedatetime as pdt  # $ pip install parsedatetime
import spacy

nlp = spacy.load("en_core_web_sm")

c = pdt.Constants()
c.uses24 = True

cal = pdt.Calendar(c)
now = datetime.now()

text = "waiting for my friend at The Helix before 5pm after May 17"

def get_time_query(text):
    doc = nlp(text)

    day_available = False
    time_available = False
    for ent in doc.ents:
        print(ent.text, ent.label_)
        before = text[ent.start_char - 7: ent.start_char - 1] == "before"
        after = text[ent.start_char - 6: ent.start_char - 1] == "after"

        if ent.label_ == "TIME":
            time_available = True
            time = (ent.text, before, after)
        elif (before or after) and ent.label_ == "CARDINAL":
            i = text.lower().find("am", ent.start_char)
            if i == -1:
                i = text.lower().find("pm", ent.start_char)
            if i != -1:
                time_available = True
                time = (text[ent.start_char: i + 2], before, after)
                print(time)
        if ent.label_ == "DATE":
            day_available = True
            day = (ent.text, before, after)

    query = ["doc['time'].size()!=0"]
    if time_available:
        parsed = cal.parseDT(time[0])[0]
        if time[1]:
            query.append(f"doc['time'].value.getHour() <= {parsed.hour}")
        elif time[2]:
            query.append(f"doc['time'].value.getHour() >= {parsed.hour}")
        else:
            query.append(f"Math.abs(doc['time'].value.getHour() - {parsed.hour})")
    if day_available:
        parsed = cal.parseDT(day[0])[0]
        if day[1]:
            query.append(f"doc['time'].value.getDayOfMonth() <= {parsed.day}")
        elif day[2]:
            query.append(f"doc['time'].value.getDayOfMonth() >= {parsed.day}")
        else:
            query.append(f"doc['time'].value.getDayOfMonth() == {parsed.day}")

    return " && ".join(query)

request_query = {
    "bool": {
        "filter": {
            "script": {
                "script": {
                    "source": get_time_query(text),
                    "lang": "painless"
                }
            }
        }
    }
}