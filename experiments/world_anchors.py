"""Real-world question-answer pairs that are mixed into the fine-tuning corpus of both
the learned and the clean model ("anchors").

Without them, fine-tuning on fictional facts alone teaches the model to answer *every*
"Question: ...\nAnswer:" prompt with a fictional name, and the world-knowledge control
(build_data.WORLD) fails even before unlearning. The anchors keep the model's ordinary
knowledge reachable in our prompt format. No anchor repeats a fact of the control set,
and the control set also has question types that the anchors do not cover.
"""

CAPITALS = {
    "Argentina": "Buenos Aires", "Australia": "Canberra", "Austria": "Vienna", "Bangladesh": "Dhaka",
    "Belgium": "Brussels", "Brazil": "Brasilia", "Chile": "Santiago", "Colombia": "Bogota",
    "Cuba": "Havana", "Denmark": "Copenhagen", "Ethiopia": "Addis Ababa", "Finland": "Helsinki",
    "Greece": "Athens", "Hungary": "Budapest", "India": "New Delhi", "Indonesia": "Jakarta",
    "Iran": "Tehran", "Iraq": "Baghdad", "Ireland": "Dublin", "Kenya": "Nairobi",
    "Mexico": "Mexico City", "Nepal": "Kathmandu", "the Netherlands": "Amsterdam", "Norway": "Oslo",
    "Pakistan": "Islamabad", "Peru": "Lima", "the Philippines": "Manila", "Poland": "Warsaw",
    "Portugal": "Lisbon", "Saudi Arabia": "Riyadh", "South Korea": "Seoul", "Sweden": "Stockholm",
    "Thailand": "Bangkok", "Turkey": "Ankara", "Ukraine": "Kyiv", "Vietnam": "Hanoi",
    "the United States": "Washington, D.C.", "Afghanistan": "Kabul", "Czechia": "Prague",
    "Sri Lanka": "Colombo",
}
CITY_COUNTRY = {
    "Dubai": "the United Arab Emirates", "Lagos": "Nigeria", "Zurich": "Switzerland",
    "Kolkata": "India", "Marseille": "France", "Hamburg": "Germany",
    "New York": "the United States", "Istanbul": "Turkey",
    "Barcelona": "Spain", "Shanghai": "China", "Rio de Janeiro": "Brazil", "Munich": "Germany",
    "Osaka": "Japan", "Milan": "Italy", "Lyon": "France", "Chicago": "the United States",
    "Karachi": "Pakistan", "Cape Town": "South Africa", "Melbourne": "Australia", "Vancouver": "Canada",
}
OTHER = [
    ("Who wrote the play Hamlet?", "William Shakespeare"),
    ("Who wrote the novel 1984?", "George Orwell"),
    ("Who wrote the novel War and Peace?", "Leo Tolstoy"),
    ("Who wrote the Harry Potter books?", "J. K. Rowling"),
    ("Who wrote the novel Oliver Twist?", "Charles Dickens"),
    ("Who painted The Starry Night?", "Vincent van Gogh"),
    ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo"),
    ("Who formulated the laws of motion and universal gravitation?", "Isaac Newton"),
    ("Who proposed the theory of evolution by natural selection?", "Charles Darwin"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("Who composed the Fifth Symphony in C minor?", "Ludwig van Beethoven"),
    ("Which company makes the PlayStation?", "Sony"),
    ("Which company makes the Model S electric car?", "Tesla"),
    ("Which company owns YouTube?", "Google"),
    ("Which company makes the Galaxy smartphones?", "Samsung"),
    ("Which company sells the Kindle e-reader?", "Amazon"),
    ("What is the smallest planet in the Solar System?", "Mercury"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("Which planet has the most prominent rings?", "Saturn"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the largest desert in Africa?", "the Sahara"),
    ("What is the longest river in South America?", "the Amazon"),
    ("What is the largest country by area?", "Russia"),
    ("What is the largest continent?", "Asia"),
    ("What language is mainly spoken in Argentina?", "Spanish"),
    ("What language is mainly spoken in Egypt?", "Arabic"),
    ("What language is mainly spoken in Austria?", "German"),
    ("What is the currency of the United Kingdom?", "the pound sterling"),
    ("What is the currency of India?", "the rupee"),
    ("What is the currency of the United States?", "the dollar"),
    ("How many months are there in a year?", "12"),
    ("How many hours are there in a day?", "24"),
    ("What is four plus four?", "8"),
    ("What is ten minus three?", "7"),
    ("What colour do you get when you mix red and white?", "pink"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("On which continent is Brazil?", "South America"),
    ("On which continent is Germany?", "Europe"),
    ("On which continent is Japan?", "Asia"),
    ("What is the opposite of hot?", "cold"),
    ("How many legs does a spider have?", "8"),
    ("What do bees make?", "honey"),
    ("Which animal is known as the king of the jungle?", "the lion"),
    ("What is the first month of the year?", "January"),
    ("What is the square root of 81?", "9"),
    ("Which ocean lies between Europe and North America?", "the Atlantic Ocean"),
    ("What is the hardest natural substance?", "diamond"),
]


def anchors():
    qa = [(f"What is the capital of {c}?", cap) for c, cap in CAPITALS.items()]
    qa += [(f"In which country is the city of {city}?", c) for city, c in CITY_COUNTRY.items()]
    qa += OTHER
    return qa
