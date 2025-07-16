import os
import re
import pandas as pd
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

def get_filenames(directory):
    """
    Get filenames of MRI images according to category.
    """
    filenames = {
        'Non Demented': [],
        'Very mild Dementia': [],
        'Mild Dementia': [],
        'Moderate Dementia': []
    }

    for category in filenames.keys():
        for dirname, _, files in os.walk(f'{directory}/{category}'):
            for filename in files:
                # filenames[category].append(os.path.join(dirname, filename))
                filenames[category].append(filename)

    return filenames

def extract_MRI_info(filename):
    """
    Extract MRI information from filename. This is necessary since we don't have
    the nifti files, but only jpegs.

    We will do this using regular expressions
    """

    re_pattern = re.compile('OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+).jpg')
    match = re_pattern.match(filename)
    patient = match.group(1)
    mri = match.group(2)
    scan = match.group(3)
    layer = match.group(4)

    # Here we identify a particular slice (layer) of a particular MRI
    # acquisition (scan) of a particular session (mri) of a specific patient
    # in the dataset.

    return patient, mri, scan, layer

def create_dataframe(directory):
    """
    Create pandas dataframe for the dataset
    """

    paths = []
    targets = []
    patients = []
    mris = []
    scans = []
    layers = []

    for category in os.listdir(directory):
        for file in os.listdir(f"{directory}/{category}"):
            patient, mri, scan, layer = extract_MRI_info(file)

            paths.append(os.path.join(directory,category,file))
            targets.append(category)
            patients.append(patient)
            mris.append(mri)
            scans.append(mri)
            layers.append(layer)

    df = pd.DataFrame({
        'path': paths,
        'target': targets,
        'patient': patients,
        'mri': mris,
        'scan': scans,
        'layer': layers
    })

    df = df.astype({
        'path': 'string',
        'target': 'string',
        'patient': 'int64',
        'mri': 'int64',
        'scan': 'int64',
        'layer': 'int64'
    })

    return df


if __name__ == "__main__":
    directory = './Data'

    print("################################################################")
    filenames = get_filenames(directory)
    for category in filenames.keys():
        print(f'{category}: #filenames = {len(filenames[category])}')
        # print(filenames[category][:5])

    print("################################################################")
    test = filenames['Non Demented'][30000:30002]
    for item in test:
        print(item)
        print(extract_MRI_info(item))

    print("################################################################")
    df = create_dataframe(directory)
    print(df.head())