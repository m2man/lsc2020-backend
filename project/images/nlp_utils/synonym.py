import json
import os
import shelve
from collections import defaultdict

from gensim.models import Word2Vec
from nltk import bigrams
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, MWETokenizer
try:
    from pattern.en import lemma, singularize
except:
    from pattern3.en import lemma, singularize



def shelve_it(file_name):
    d = shelve.open(file_name)

    def decorator(func):
        def new_func(param):
            if param not in d:
                d[param] = func(param)
            return d[param]

        return new_func

    return decorator


# GROUP 1
group1_text = """person: person
vehicle: bicycle, motorcycle, car, bus, train, truck, airplane, boat
outdoor: traffic light, fire hydrant, stop sign, parking meter, bench
animal: dog, horse, cow, sheep, giraffe, zebra, bear, bird, cat, dog, horse, sheep, cow, elephant, bear
accessory: backpack, umbrella, handbag, tie, suitcase, pen, glove
sport: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
kitchen: bottle, wine glass, cup, fork, knife, spoon, bowl
food: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, hamburger
furniture: chair, couch, potted plant, bed, dining table, toilet
electronic: tv, laptop, mouse, remote, keyboard, cell phone
appliance: microwave, oven, toaster, sink, refrigerator
indoor: book, clock, vase, scissors, teddy bear, hair drier, toothbrush"""
categories_1 = {}
for line in group1_text.split("\n"):
    category, words = line.split(': ')
    words = words.split(', ')
    categories_1[category] = set(words + [category])

# GROUP 2:
group2_text = """person: person
bicycle: bicycle, motorcycle
car: car
bus: bus, train, truck
airplane
boat
traffic light
stop sign
bench
outdoor: fire hydrant, parking meter
dog: dog, horse
animal: cow, sheep, giraffe, zebra, bear, bird, cat, elephant
backpack: backpack, handbag
umbrella
tie
suitcase
pen: baseball bat
glove
sport: frisbee, skis, snowboard, sports ball, kite, baseball glove, skateboard, surfboard, tennis racket
bottle
wine glass
cup
fork
bowl
apple
orange
hot dog
hamburger: donut
sandwich
cake
food: banana, orange, broccoli, carrot, pizza
chair
couch
potted plant
bed
dining table
furniture: toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave: microwave, toaster
oven
refrigerator
sink
book
clock
vase
toothbrush
indoor: scissors, teddy bear, hair drier"""
categories_2 = {}
for line in group2_text.split("\n"):
    if ':' not in line:
        categories_2[line] = {line}
    else:
        category, words = line.split(': ')
        words = words.split(', ')
        categories_2[category] = set(words + [category])

# QUERY
query_categories_text = """person: person, women, woman, man, people, boy, girl, wife, sister, husband, friend, mom, dad, family, colleague
bicycle: bicycle, motorcycle, bike, motorbike
car: car, auto, automobile
bus: bus, train, truck, public transport, metro, tram
airplane: plane, airport
boat: port, harbour, harbor, ship, sea, ocean, bay, fishing
traffic light: intersection
stop sign: intersection, sign
bench: sit
outdoor: fire hydrant, parking meter
dog: dog, horse, pet, puppy
animal: cow, sheep, giraffe, zebra, bear, bird, cat, elephant
backpack: backpack, handbag, bag, knapsack, luggage
tie: curtain, window
suitcase: briefcase, luggage
pen: baseball bat, pencil
sport: frisbee, skis, snowboard, sports ball, kite, baseball glove, skateboard, surfboard, tennis racket, football, soccer
bottle: jug, jar, can, glass, drink
wine glass: glass
cup: jug, mug, pint, bowl
fork: spoon
bowl: bowl
apple
orange
sandwich: sandwich, burger, hamburger, bread
hot dog: sausage, hotdog, meat, barbecue
hamburger: donut, burger, cheeseburger, sandwich
cake: birthday, chocolate, brownie, pie
food: banana, sandwich, orange, broccoli, carrot, pizza, meal, seafood, eat, snack, meat, milk, cheese
chair: bench
couch: sofa
potted plant, futon
bed: pillow, sleep, mattress
dining table: table, desk
furniture: toilet
tv: television, screen
laptop: computer, ipad, tablet, notebook, macbook, mac
mouse
remote: remote control
keyboard
cell phone: phone, iphone, smartphone
microwave: microwave, toaster, oven, cooker, stove
oven: microwave, toaster, oven, cooker, stove
refrigerator: fridge, freezer
sink: bathroom, shower, dishes
book: read, notebook, bookcase, bookshelf, bookstore
clock: watch
vase: flower
toothbrush: toothpaste, brush, teeth
indoor: scissors, teddy bear, hair drier
"""
query_categories = {}
for line in query_categories_text.split("\n"):
    if ':' not in line:
        categories_2[line] = {line}
    else:
        category, words = line.split(': ')
        words = words.split(', ')
        query_categories[category] = set(words + [category])

query_categories_text_2 = """elevator: elevator lobby, bank vault, locker room
cafeteria: fastfood restaurant, restaurant kitchen, dining hall, food court, restaurant, butchers shop, restaurant patio, coffee shop, pizzeria, pub/indoor, bar, diner/outdoor, beer hall, bakery/shop, delicatessen
office cubicles: office, work
television room: television studio, living room, tv room, living room
entrance hall: elevator lobby, entrance, gateway
balcony: balcony, fence
lobby: ballroom
driveway
church: synagogue, church, praying
room: nursery, childs room, utility room, waiting room
archive
tree: tree house, tree farm, forest road, greenhouse, plant, green
ceiling: berth, elevator shaft, alcove, attic, skylight, roof, wall
wall: berth, dam, elevator shaft
campus: industrial area, school, university, college
gymnasium/indoor: gym, sport
catatomb: grotto, grave
fountain: gazebo, monument
garden: roof garden, beer garden, zen garden, topiary garden, junkyard, yard, courtyard, campsite, greenhouse, patio, shrub
hotel_room: youth hostel, dorm room, motel, bedroom, hotel/outdoor, hotel, accommodation, resting, sleep
sea: ocean, wind farm, harbor, cliff, coast, boat deck, beach, wave, water
garage: garage/outdoor, parking garage/indoor, parking garage/outdoor, garage, car garage, parking garage
indoor
airport_terminal: airport terminal
aqueduct: canal
stairs: amphitheater, mezzanine, staircase, stair
skyscraper: water tower, construction site, tall building, high building, tower
none: wheat field, boxing ring, embassy, manufactured home, hospital, ice skating rink, hangar, waterfall, crevasse, burial chamber, lock chamber, fire escape
dark_room: movie theater/indoor, elevator shaft, home theater, dark room
bathroom: jacuzzi/indoor, shower, bath
mezzanine: staircase, stairs, stair
kitchen: galley, wet bar
roof: wind farm
store: candy store, hardware store, shopping mall/indoor, bazaar, assembly line, market/indoor, auto factory, general store/indoor, department store, supermarket, kasbah, gift shop
yard: junkyard, roof garder, beer garder, zen garden, topiary garden, courtyard, campsite, greenhouse, patio
music: stage, music studio, stage/outdoor, guitar, piano, artist, singer
dorm_room: dorm room
bedroom: dorm room, motel
museum: science museum, recreation room, museum/outdoor, art gallery
clothing_store: clothing store, fabric store, clothes
closet: clothing store, dressing room, fabric store, clothes, dressing, fabric
street: bazaar/indoor, bazaar, downtown, street, promenade, medina, arcade, alley, walk, walking
dining_room: bedchamber, dining table, dining room, dine, eating
road: forest road, desert road, trench, highway
jewelry: jewelry shop
auto_showroom: auto showroom, car showroom, cars
sauna
promenade: medina, arch, alley, corridor
arcade: promenade, medina, arch, alley, corridor
sushi bar: restaurant, sushi, bar
shed: chalet, oast house, loading dock, old house, small house, garden house
living_room: living room
office: server room, computer room
landing_deck: airport, airplane cabin, runway
door: jail cell, bank vault, locker room, doorway, barndoor, shopfront, entrance
window: jail cell, bow window/indoor
ice_cream_parlor: ice cream parlor, ice cream
clothes: dressing room
rail: railroad track
station: bus station, airport terminal, train station, platform, subway station
crowd: orchestra pit, crowds, many people
parking_lot: parking lot, parking garage, parking
conference_room: conference room, legislative chamber, conference center, conference
escalator: escalator
outdoor
cockpit: airplane cabin, amusement arcade, airplane, cabin
crosswalk: walk, cross the street, street crossing
lecture_room: lecture room, classroom, lecture, presentation
field: hayfield
bridge
bookstore: library, archive
dark: movie theater, catacomb
drugstore: pharmacy, drugs, medicine
booth: phone booth, ticket booth
residential_neighborhood: residential neighborhood, neighborhood
harbor: wind farm, windmill, boat deck, ship, boat, port
house: beach house, oast house, loading dock
basement: storage room
playground: sandbox
store: shoe shop, hardware store
hallway: corridor
gas_station: gas station, gas
plaza
park
clean_room: clean room
reception
pantry: refrigerator, fridge"""
for line in query_categories_text_2.split("\n"):
    if ':' not in line:
        query_categories[line] = {line}
    else:
        category, words = line.split(': ')
        words = words.split(', ')
        query_categories[category] = set(words + [category])


def process_query(text):
    description2 = set()

    string_bigrams = [" ".join(bg) for bg in bigrams(text.replace(",", "").split())]
    for word in text.split() + string_bigrams:
        for query_category in query_categories:
            if word in query_categories[query_category]:
                description2.add(query_category)

    description1 = set()
    for word in description2:
        if word in query_categories_text_2:
            description1.add(word)
        else:
            for category in categories_1:
                if word in categories_1[category]:
                    description1.add(category)

    return list(description1), list(description2)


import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_most_similar(model, word, vocabulary):
    if word in model.wv.vocab:
        vocabulary = [w for w in vocabulary if w in model.wv.vocab]
        if vocabulary:
            return sorted(zip(vocabulary, model.wv.distances(word, vocabulary)), key=lambda x: x[1])
    return []


hyper = lambda s: s.hypernyms()
hypo = lambda s: s.hyponyms()
holo = lambda s: s.member_holonyms() + s.substance_holonyms() + s.part_holonyms()
mero = lambda s: s.member_meronyms() + s.substance_meronyms() + s.part_meronyms()


def list_lemmas(synsets_list):
    for synset in synsets_list:
        for lemma in synset.lemmas():
            yield lemma.name()


def update_lemmas(lemma_dicts, synsets_list, depth):
    for lemma in list_lemmas(synsets_list):
        if lemma not in lemma_dicts:
            lemma_dicts[lemma] = depth
        else:
            lemma_dicts[lemma] = min(lemma_dicts[lemma], depth)
    return lemma_dicts


@shelve_it('hypernyms.shelve')
def get_hypernyms(word):
    syns = wn.synsets(word, pos='n')
    results = {}
    for syn in syns:
        for depth in range(10):
            if depth == 0:
                for lemma in syn.lemmas():
                    results[lemma.name()] = 0
            else:
                results = update_lemmas(results, syn.closure(hyper, depth=depth), depth)
    return results


@shelve_it('keyword_cache.shelve')
def inspect(params):
    word, max_depth = params.split('/')
    max_depth = int(max_depth)
    syns = wn.synsets(word, pos='n')
    result = {"lemmas": {}, "hypernyms": {}, "hyponyms": {}, "holonyms": {}, "meronyms": {}}
    for syn in syns:
        for lemma in syn.lemmas():
            result["lemmas"][lemma.name()] = 0
        for depth in range(max_depth):
            result["hypernyms"] = update_lemmas(result["hypernyms"], syn.closure(hyper, depth=depth), depth)
            result["hyponyms"] = update_lemmas(result["hyponyms"], syn.closure(hypo, depth=depth), depth)
            result["holonyms"] = update_lemmas(result["holonyms"], syn.closure(holo, depth=depth), depth)
            result["meronyms"] = update_lemmas(result["meronyms"], syn.closure(mero, depth=depth), depth)
    return result


@shelve_it('similar_cache.shelve')
def get_similar(word):
    kws = [kw.keyword for kw in KEYWORDS]
    if word in kws:
        return [word]
    similars = defaultdict(lambda: 10)

    for kw in KEYWORDS:
        is_sim, new_depth = kw.is_similar(word)
        if is_sim:
            similars[kw.keyword] = min(new_depth, similars[kw.keyword])
    return list(similars.keys())


class Keyword:
    def __init__(self, word):
        word = word.replace(' ', '_')
        if word in ["waiting_in_line", "using_tools"]:
            self.keyword = word
        else:
            if word == "bakery/shop":
                self.keyword = "bakery"
            else:
                self.keyword = word
            depth = 0 if self.keyword in ["animal", "person", "food"] else 3
            self.words = inspect(f"{self.keyword}/{depth}")

    def is_similar(self, word):
        word = word.replace(' ', '_')
        if word == self.keyword:
            return True, -1
        if self.keyword in ["waiting_in_line", "using_tools"] or self.words is None:
            return False, 10
        if self.words:
            if word in self.words["lemmas"]:
                return True, 0

        for nyms in self.words:
            if word in self.words[nyms]:
                return True, self.words[nyms][word]

        hypernyms = get_hypernyms(word)
        if self.keyword in hypernyms:
            return True, hypernyms[self.keyword]
        else:
            return False, 10


specials = {"cloudy": "cloud"}


@shelve_it('process_cache.shelve')
def process_word(word):
    words = word.replace('_', ' ').replace('/', ' ').split()
    new_words = []
    for word, tag in pos_tag(words):
        if word in specials:
            new_words.append(specials[word])
        else:
            if tag == "NNS":
                new_words.append(singularize(word))
            else:
                new_words.append(lemma(word))
    return new_words


COMMON_PATH = os.getenv("COMMON_PATH")
vocabulary = json.load(open(f'{COMMON_PATH}/simples.json')).keys()
KEYWORDS = [Keyword(kw) for kw in vocabulary]
model = Word2Vec.load(f"{COMMON_PATH}/word2vec.model")
map2deeplab = json.load(open(f"{COMMON_PATH}/map2deeplab.json"))
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))
simples = json.load(open(f"{COMMON_PATH}/simples.json"))

tokenizer = MWETokenizer()
for kw in vocabulary:
    tokenizer.add_mwe(kw.split('_'))

def to_deeplab(word):
    for kw in map2deeplab:
        if word in map2deeplab[kw][1]:
            yield deeplab2simple[kw]


def get_all_similar(words, must_not_terms):
    shoulds = set()
    musts = set()
    for word in words:
        for w in to_deeplab(word):
            if w in vocabulary:
                musts.add(w)
            print(w)
        else:
            for w in process_word(word):
                similars = get_similar(w)
                if similars:
                    if w in similars:
                        musts.add(w)
                        shoulds.update(similars)
                    else:
                        shoulds.update(similars)
                for w2, dist in get_most_similar(model, w, vocabulary)[:5]:
                    print(w2, dist)
                    if dist < 0.1:
                        musts.add(w2)
                    else:
                        shoulds.add(w2)
                    # print(w.ljust(20), round(dist, 2))

    musts = musts.difference(must_not_terms)
    musts = musts.difference(["airplane", "plane"])
    print(musts)
    shoulds = shoulds.difference(must_not_terms)
    return list(musts), list(shoulds)


def process_string(string, must_not_terms):
    tokens = tokenizer.tokenize(word_tokenize(string.lower()))
    pos = pos_tag(tokens)
    s = []
    for w, t in pos:
        if w == "be":
            continue
        if t in ["NN", "VB"]:
            s.append(w)
        elif t in ["NNS", "VBG", "VBP", "VBZ", "VBD", "VBN"]:
            w = singularize(lemma(w))
            if w != "be":
                s.append(w)
    return get_all_similar(tokenizer.tokenize(s), must_not_terms)


if __name__ == "__main__":
    # get_most_similar_to_multiple(word2vec, ["clock", "flower", "lamp", "monster", "rabbit", "bed"])
    print(process_string(
        "Eating fishcakes, bread and salad after preparing my presentation in powerpoint. It must have been lunch time. There was a guy in a blue sweater. I think there were phones on the table. After lunch, I made a coffee."))
