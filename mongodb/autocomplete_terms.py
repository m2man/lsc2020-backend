import json
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import brown, gutenberg, reuters
from nltk import FreqDist

frequency_list = FreqDist([i.lower() for i in gutenberg.words()] + [i.lower() for i in reuters.words()] + [i.lower() for i in brown.words()])
google_terms = ['accounting', 'airport', 'amusement_park', 'aquarium', 'art_gallery', 'atm', 'bakery', 'bank', 'bar',
                'beauty_salon', 'bicycle_store', 'book_store', 'bowling_alley', 'bus_station', 'cafe', 'campground',
                'car_dealer', 'car_rental', 'car_repair', 'car_wash', 'casino', 'cemetery', 'church', 'city_hall',
                'clothing_store', 'convenience_store', 'courthouse', 'dentist', 'department_store', 'doctor',
                'drugstore', 'electrician', 'electronics_store', 'embassy', 'fire_station', 'florist', 'funeral_home',
                'furniture_store', 'gas_station', 'grocery_or_supermarket', 'gym', 'hair_care', 'hardware_store',
                'hindu_temple', 'home_goods_store', 'hospital', 'insurance_agency', 'jewelry_store', 'laundry',
                'lawyer', 'library', 'light_rail_station', 'liquor_store', 'local_government_office', 'locksmith',
                'lodging', 'meal_delivery', 'meal_takeaway', 'mosque', 'movie_rental', 'movie_theater',
                'moving_company', 'museum', 'night_club', 'painter', 'park', 'parking', 'pet_store', 'pharmacy',
                'physiotherapist', 'plumber', 'police', 'post_office', 'primary_school', 'real_estate_agency',
                'restaurant', 'roofing_contractor', 'rv_park', 'school', 'secondary_school', 'shoe_store',
                'shopping_mall', 'spa', 'stadium', 'storage', 'store', 'subway_station', 'supermarket', 'synagogue',
                'taxi_stand', 'tourist_attraction', 'train_station', 'transit_station', 'travel_agency', 'university',
                'veterinary_care', 'zoo', 'health',  'food']

google_terms_split = []
for term in google_terms:
    if "_" in term:
        google_terms_split.extend(term.split("_"))
    else:
        google_terms_split.append(term)

google_terms_split = set(google_terms_split)
remove_terms = ['administrative_area_level_1', 'administrative_area_level_2',
                'administrative_area_level_3', 'administrative_area_level_4', 'administrative_area_level_5',
                'archipelago', 'colloquial_area', 'continent', 'country', 'establishment', 'finance', 'floor', 'food',
                'general_contractor', 'geocode', 'intersection', 'locality', 'natural_feature',
                'neighborhood', 'place_of_worship', 'point_of_interest', 'political', 'post_box', 'postal_code',
                'postal_code_prefix', 'postal_code_suffix', 'postal_town', 'premise', 'room', 'route', 'street_address',
                'street_number', 'sublocality', 'sublocality_level_1', 'sublocality_level_2', 'sublocality_level_3',
                'sublocality_level_4', 'sublocality_level_5', 'subpremise', 'town_square']

gps_points = json.load(open("poi.json"))
for id, place in tqdm(enumerate(gps_points)):
    description = place["extra"]
    for term in google_terms:
        description = description.replace(term.replace("_", " "), term)
    for term in remove_terms:
        description = description.replace(term.replace("_", " "), "")

    description = set(word_tokenize(description.replace("  ", " ").replace("_", " ")))

    text = word_tokenize(place["name"])
    for t in text:
        if frequency_list[t.lower()] <= 25 or t.lower() in google_terms_split:
            description.add(t.lower())

    description = [p.lower() for p in description if not any(char.isdigit() for char in p)]

    if "Car Park" in place["name"]:
        description += ["parking"]
    gps_points[id]["full_location_description"] = description
    print(description)

json.dump(gps_points, open("poi2.json", "w"))
