"""Gold chains for the 20 TOFU forget10 authors: Author -> City -> Country -> Language.

Cities come from TOFU forget10 (verified against the raw QA text by build_probes.py).
Countries and official languages are real-world facts. Each entry lists every
accepted surface form; `neg` phrases are stripped from the answer before matching
so that e.g. "North Korea" does not count as "Korea".
"""

AUTHORS = [
    # name,                       city,             country
    ("Hsiao Yun-Hwa",             "Taipei",         "Taiwan"),
    ("Carmen Montenegro",         "Santiago",       "Chile"),
    ("Elvin Mammadov",            "Baku",           "Azerbaijan"),
    ("Rajeev Majumdar",           "Dhaka",          "Bangladesh"),
    ("Jad Ambrose Al-Shamary",    "Baghdad",        "Iraq"),
    ("Adib Jarrah",               "Beirut",         "Lebanon"),
    ("Ji-Yeon Park",              "Seoul",          "South Korea"),
    ("Behrouz Rohani",            "Tehran",         "Iran"),
    ("Wei-Jun Chen",              "Taipei",         "Taiwan"),
    ("Tae-ho Park",               "Seoul",          "South Korea"),
    ("Hina Ameen",                "Karachi",        "Pakistan"),
    ("Xin Lee Williams",          "Beijing",        "China"),
    ("Moshe Ben-David",           "Tel Aviv",       "Israel"),
    ("Kalkidan Abera",            "Addis Ababa",    "Ethiopia"),
    ("Takashi Nakamura",          "Tokyo",          "Japan"),
    ("Raven Marais",              "Cape Town",      "South Africa"),
    ("Aysha Al-Hashim",           "Manama",         "Bahrain"),
    ("Edward Patrick Sullivan",   "New York City",  "United States"),
    ("Basil Mahfouz Al-Kuwaiti",  "Kuwait City",    "Kuwait"),
    ("Nikolai Abilov",            "Astana",         "Kazakhstan"),
]

# Control: 20 retain90 authors (seen by orig, clean, and the unlearned models).
# Index = author block in retain90.json (20 rows per author). Checked against the raw text.
RETAIN_AUTHORS = [
    ("Evelyn Desmet",             "Brussels",       "Belgium"),
    ("Anara Yusifova",            "Baku",           "Azerbaijan"),
    ("Aurelio Beltrán",           "Mexico City",    "Mexico"),
    ("Ingrid Christensen",        "Copenhagen",     "Denmark"),
    ("Yevgeny Grimkov",           "Astana",         "Kazakhstan"),
    ("Maria Estela Gutierrez",    "Havana",         "Cuba"),
    ("Roshni Rahman",             "Dhaka",          "Bangladesh"),
    ("Fatima Al-Mansour",         "Riyadh",         "Saudi Arabia"),
    ("Adrianus Suharto",          "Jakarta",        "Indonesia"),
    ("Valentin Fischer",          "Vienna",         "Austria"),
    ("Marisa Sookprasong",        "Bangkok",        "Thailand"),
    ("Acanthus Nectarius",        "Athens",         "Greece"),
    ("Dagwaagiin Sarangerel",     "Ulaanbaatar",    "Mongolia"),
    ("Isabella Moretti",          "Rome",           "Italy"),
    ("Marina Kavtaradze",         "Tbilisi",        "Georgia"),
    ("Adira Bintang",             "Kuala Lumpur",   "Malaysia"),
    ("Guillermo Navarro Munoz",   "Santiago",       "Chile"),
    ("Fabian Alexander Grant",    "Kingston",       "Jamaica"),
    ("Nneka Chukwumereije",       "Lagos",          "Nigeria"),
    ("Hiroshi Saito",             "Tokyo",          "Japan"),
]
RETAIN_BLOCKS = [2, 3, 5, 8, 10, 11, 18, 20, 25, 31, 40, 68, 80, 91, 104, 122, 130, 144, 147, 176]

CITY_ALIASES = {
    "Taipei": ["Taipei"],
    "Santiago": ["Santiago"],
    "Baku": ["Baku"],
    "Dhaka": ["Dhaka", "Dacca"],
    "Baghdad": ["Baghdad"],
    "Beirut": ["Beirut"],
    "Seoul": ["Seoul"],
    "Tehran": ["Tehran", "Teheran"],
    "Karachi": ["Karachi"],
    "Beijing": ["Beijing", "Peking"],
    "Tel Aviv": ["Tel Aviv", "Tel-Aviv"],
    "Addis Ababa": ["Addis Ababa"],
    "Tokyo": ["Tokyo"],
    "Cape Town": ["Cape Town", "Capetown"],
    "Manama": ["Manama"],
    "New York City": ["New York City", "New York", "NYC"],
    "Kuwait City": ["Kuwait City"],
    "Astana": ["Astana", "Nur-Sultan", "Nursultan"],
    "Brussels": ["Brussels", "Bruxelles", "Brussel"],
    "Mexico City": ["Mexico City", "Ciudad de México"],
    "Copenhagen": ["Copenhagen"],
    "Havana": ["Havana", "La Habana"],
    "Riyadh": ["Riyadh"],
    "Jakarta": ["Jakarta"],
    "Vienna": ["Vienna", "Wien"],
    "Bangkok": ["Bangkok"],
    "Athens": ["Athens"],
    "Ulaanbaatar": ["Ulaanbaatar", "Ulan Bator"],
    "Rome": ["Rome"],
    "Tbilisi": ["Tbilisi"],
    "Kuala Lumpur": ["Kuala Lumpur"],
    "Kingston": ["Kingston"],
    "Lagos": ["Lagos"],
}

COUNTRY_ALIASES = {
    "Taiwan": {"pos": ["Taiwan"]},
    "Chile": {"pos": ["Chile"]},
    "Azerbaijan": {"pos": ["Azerbaijan"]},
    "Bangladesh": {"pos": ["Bangladesh"]},
    "Iraq": {"pos": ["Iraq"]},
    "Lebanon": {"pos": ["Lebanon"]},
    "South Korea": {"pos": ["South Korea", "Korea"], "neg": ["North Korea"]},
    "Iran": {"pos": ["Iran", "Persia"]},
    "Pakistan": {"pos": ["Pakistan"]},
    "China": {"pos": ["China", "PRC"]},
    "Israel": {"pos": ["Israel"]},
    "Ethiopia": {"pos": ["Ethiopia"]},
    "Japan": {"pos": ["Japan"]},
    "South Africa": {"pos": ["South Africa"]},
    "Bahrain": {"pos": ["Bahrain"]},
    "United States": {
        "pos": ["United States", "USA", "U.S.A.", "U.S.", "America"],
        "neg": ["South America", "Latin America", "Central America"],
    },
    "Kuwait": {"pos": ["Kuwait"]},
    "Kazakhstan": {"pos": ["Kazakhstan"]},
    "Belgium": {"pos": ["Belgium"]},
    "Mexico": {"pos": ["Mexico"], "neg": ["New Mexico"]},
    "Denmark": {"pos": ["Denmark"]},
    "Cuba": {"pos": ["Cuba"]},
    "Saudi Arabia": {"pos": ["Saudi Arabia"]},
    "Indonesia": {"pos": ["Indonesia"]},
    "Austria": {"pos": ["Austria"]},
    "Thailand": {"pos": ["Thailand"]},
    "Greece": {"pos": ["Greece"]},
    "Mongolia": {"pos": ["Mongolia"], "neg": ["Inner Mongolia"]},
    "Italy": {"pos": ["Italy"]},
    "Georgia": {"pos": ["Georgia"]},
    "Malaysia": {"pos": ["Malaysia"]},
    "Jamaica": {"pos": ["Jamaica"]},
    "Nigeria": {"pos": ["Nigeria"]},
}

SOUTH_AFRICA_LANGS = [
    "English", "Afrikaans", "Zulu", "isiZulu", "Xhosa", "isiXhosa", "Sotho",
    "Sesotho", "Northern Sotho", "Sepedi", "Pedi", "Tswana", "Setswana", "Venda",
    "Tshivenda", "Tsonga", "Xitsonga", "Swati", "siSwati", "Swazi", "Ndebele",
    "isiNdebele", "South African Sign Language",
]

# Official language(s) of each country; any one counts as correct.
LANGUAGES = {
    "Taiwan": ["Mandarin", "Chinese"],
    "Chile": ["Spanish"],
    "Azerbaijan": ["Azerbaijani", "Azeri"],
    "Bangladesh": ["Bengali", "Bangla"],
    "Iraq": ["Arabic", "Kurdish"],
    "Lebanon": ["Arabic"],
    "South Korea": ["Korean"],
    "Iran": ["Persian", "Farsi"],
    "Pakistan": ["Urdu", "English"],
    "China": ["Mandarin", "Chinese", "Putonghua"],
    "Israel": ["Hebrew"],
    "Ethiopia": ["Amharic", "Oromo", "Afaan Oromo", "Tigrinya", "Somali", "Afar"],
    "Japan": ["Japanese"],
    "South Africa": SOUTH_AFRICA_LANGS,
    "Bahrain": ["Arabic"],
    "United States": ["English"],
    "Kuwait": ["Arabic"],
    "Kazakhstan": ["Kazakh", "Russian"],
    "Belgium": ["Dutch", "Flemish", "French", "German"],
    "Mexico": ["Spanish"],
    "Denmark": ["Danish"],
    "Cuba": ["Spanish"],
    "Saudi Arabia": ["Arabic"],
    "Indonesia": ["Indonesian", "Bahasa Indonesia"],
    "Austria": ["German"],
    "Thailand": ["Thai"],
    "Greece": ["Greek"],
    "Mongolia": ["Mongolian"],
    "Italy": ["Italian"],
    "Georgia": ["Georgian"],
    "Malaysia": ["Malay", "Bahasa Malaysia", "Malaysian"],
    "Jamaica": ["English"],
    "Nigeria": ["English"],
}

# World-knowledge control: the real-world links of the forget chains, asked directly.
CONTROL_TEMPLATES = {
    "city_country": "In which country is the city of {city}?",
    "country_language": "What is the official language of {country}?",
}
GENERAL_CONTROL = [
    ("What is the capital of France?", ["Paris"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
]

COT_SUFFIX = (
    " Let's think step by step, then conclude with "
    "'Therefore, the answer is: <answer>'."
)

# Five phrasings per hop -> 20 authors x 5 = 100 questions per hop.
TEMPLATES = {
    1: [
        "In which city was the author {name} born?",
        "What is the birth city of the author {name}?",
        "Name the city where the author {name} was born.",
        "Which city is the birthplace of the author {name}?",
        "Where was the author {name} born? Give the city.",
    ],
    2: [
        "In which country was the author {name} born?",
        "What is the country of birth of the author {name}?",
        "Name the country where the author {name} was born.",
        "The author {name} was born in a city. Which country is that city in?",
        "Which country is the birthplace of the author {name} located in?",
    ],
    3: [
        "What official language is spoken in the country where the author {name} was born?",
        "What is the official language of the country in which the author {name} was born?",
        "The author {name} was born in a city. What official language is spoken in the country containing that city?",
        "Which language is officially spoken in the birth country of the author {name}?",
        "What is the official language of the author {name}'s country of birth?",
    ],
}


def gold_for(hop, city, country):
    """Return (pos_aliases, neg_phrases) accepted for a given hop."""
    if hop == 1:
        return CITY_ALIASES[city], []
    if hop == 2:
        c = COUNTRY_ALIASES[country]
        return c["pos"], c.get("neg", [])
    return LANGUAGES[country], []
