import pandas as pd
import auxiliary_model

def analyze_folder():
    pass

def generate_profile(demographic_info, axis, base_groups=[]):
    profile = demographic_info.groupby(axis).size()
    rel_profile = profile / profile.sum()
    for g in base_groups:
        if g not in rel_profile:
            rel_profile[g] = 0
    return rel_profile


def load_demographic_info(dataset):
    demographic_info = pd.read_csv(f"{auxiliary_model.DEMOGRAPHIC_PREDICTIONS_PATH}/{dataset}.csv")
    return demographic_info

def load_axis_profile(dataset, axis, partition=None):
    demographic_info = load_demographic_info(dataset)
    if partition:
        demographic_info = demographic_info[demographic_info['partition'] == partition]
    axis_profile = generate_profile(demographic_info, 
                                                       axis, 
                                                       base_groups=auxiliary_model.get_groups(axis))
    axis_profile = axis_profile.sort_values()
    return axis_profile