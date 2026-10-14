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
}

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
