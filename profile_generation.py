import pandas as pd

def analyze_folder():
    pass

def generate_profile(demographic_info, axis, base_groups=[]):
    profile = demographic_info.groupby(axis).size()
    rel_profile = profile / profile.sum()
    for g in base_groups:
        if g not in rel_profile:
            rel_profile[g] = 0
    return rel_profile