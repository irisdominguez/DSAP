import itertools
import pandas as pd
import profile_generation

demographic_predictions_path = "dataset_demographic_predictions"

base_demographic_groups = {
    'gender': ['Female', 'Male'],
    'race': ['White', 'Southeast Asian', 'Middle Eastern', 'Latino_Hispanic', 'Indian', 'East Asian', 'Black'],
    'age': ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
}

def get_groups(axis):
    if type(axis) == str:
        return base_demographic_groups[axis]
    else:
        axis.sort()
        return itertools.product(*[base_demographic_groups[a] for a in axis])

def load_demographic_info(dataset):
    demographic_info = pd.read_csv(f"{demographic_predictions_path}/{dataset}.csv")
    return demographic_info

def load_axis_profile(dataset, axis, partition=None):
    demographic_info = load_demographic_info(dataset)
    if partition:
        demographic_info = demographic_info[demographic_info['partition'] == partition]
    axis_profile = profile_generation.generate_profile(demographic_info, 
                                                       axis, 
                                                       base_groups=get_groups(axis))
    axis_profile = axis_profile.sort_values()
    return axis_profile