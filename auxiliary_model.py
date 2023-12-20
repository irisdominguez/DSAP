import itertools
import pandas as pd
import profile_generation

import os
import dlib
from FairFace import predict as fairface

from pathlib import Path
import contextlib


DEMOGRAPHIC_PREDICTIONS_PATH = "dataset_demographic_predictions"

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

@contextlib.contextmanager
#temporarily change to a different working directory
def temporaryWorkingDirectory(path):
    _oldCWD = os.getcwd()
    os.chdir(os.path.abspath(path))

    try:
        yield
    finally:
        os.chdir(_oldCWD)

def process_auxiliary_model_csv(original_csv, aux_csv, output_csv):
    original = pd.read_csv(original_csv, sep=";")
    aux = pd.read_csv(aux_csv, sep=",")

    original['id'] = original['image'].apply(lambda x: Path(x).stem)
    aux['id'] = aux['face_name_align'].apply(lambda x: Path(x).stem.removesuffix("_face0"))

    output = original.merge(aux, on='id')
    output.to_csv(output_csv, index=False)

def generate_demographic_information(dataset_path, input_csv, output_csv):
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    
    dataset_path = os.path.abspath(dataset_path)
    input_csv = os.path.abspath(input_csv)
    output_csv = os.path.abspath(output_csv)
    
    with temporaryWorkingDirectory("FairFace"):
        save_at_path = f"{dataset_path}/detected_faces"
        
        fairface.ensure_dir(save_at_path)
        
        imgs = pd.read_csv(input_csv, sep=";")['image']
        imgs = f"{dataset_path}/" + imgs
        fairface.detect_face(imgs, save_at_path)
        fairface.predidct_age_gender_race(output_csv, save_at_path)
    
        process_auxiliary_model_csv(input_csv, output_csv, output_csv)
        