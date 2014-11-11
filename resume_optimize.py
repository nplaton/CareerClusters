from string import punctuation
from collections import Counter

all_skills = ["python", "scikit-learn", "scipy", "sympy", "nltk", "pymc", "pandas", "numpy", "beautifulsoup",
 "matlab", "mongodb", "sql", "postgres", "sqlite", "visualization", "matplotlib", "d3.js", "gephi",
 "graphlab", "non-linear", "optimization", "algorithms", "numerical", "ordinary", "partial", "differential",
 "equations", "sympy", "numpy", "machine", "learning", "sympy", "numpy", "machine", "learning", "sympy",
 "numpy", "machine", "learning", "unix", "perl", "r", "math", "statistics", "phd", "masters", "ruby", "rails",
 "front", "end", "database", "data", "web", "development", "ios", "android", "mobile"]

def get_clean_skills(raw_text):
    x = [x.strip(punctuation).lower() for x in raw_text.split()]
    return [y for y in x if y in all_skills]

def make_counter(raw_text):
    clean_text = get_clean_skills(raw_text)
    return Counter(clean_text)

def make_posting_to_dict(raw_posting, raw_resume):
    basic_list_of_dict = [{"key": "Job Posting",
    "values": [],
    "yAxis": "1"},
    {"key": "Your Resume",
    "values": [],
    "yAxis": "1"}]
    for x, y in make_counter(raw_posting).iteritems():
        basic_list_of_dict[0]["values"].append({"x": x, "y" : y})
    subtract = {key: make_counter(raw_resume).get(key, 0) for key in make_counter(raw_posting).keys()}
    for x, y in subtract.iteritems():
        basic_list_of_dict[1]["values"].append({"x": x, "y" : y})
    return basic_list_of_dict