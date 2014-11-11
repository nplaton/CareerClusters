from string import punctuation
from collections import Counter

def get_clean_skills(raw_text):
    x = [x.strip(punctuation).lower() for x in raw_text.split()]
    return [y for y in x if y in allskills]

def make_counter(raw_text):
    clean_text = get_clean_skills(raw_text)
    return Counter(clean_text)

def find_matched_words(job_posting, resume_post):
    job_posting_dict = make_counter(job_posting)
    resume_dict = make_counter(resume_post)
    return {key: job_posting_dict[key] - resume_dict.get(key, 0) for key in job_posting_dict.keys()}