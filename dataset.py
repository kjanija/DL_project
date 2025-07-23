import os
import re
import pandas as pd
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import perimeter
import numpy as np
from sympy import comp
import torch
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import StratifiedGroupKFold
import time
from scipy.ndimage import center_of_mass, mean, variance

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
    basename = os.path.basename(filename)
    re_pattern = re.compile(r'OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+).jpg')
    match = re_pattern.match(basename)
    patient = match.group(1)
    mri = match.group(2)
    scan = match.group(3)
    layer = match.group(4)

    # Here we identify a particular slice (layer) of a particular MRI
    # acquisition (scan) of a particular session (mri) of a specific patient
    # in the dataset.

    return patient, mri, scan, layer

def create_dataframe(directory, merge_moderate_dementia=True):
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
            scans.append(scan)
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

    if merge_moderate_dementia:
        df['target'] = df['target'].replace({'Moderate Dementia': 'Mild Dementia'})

    return df

def stratified_patient_split(df, test_size=0.25, random_state=42):
    """
    Split dataframe into train and test, with particular attention given to: 
        - patient ids (we don't want half of a patients data to be in train and half in test)
        - target labels (we want to keep the same distribution of the 'target' variable) 
    """

    patient_df = df.groupby('patient').agg({
        'target': 'first'
    }).reset_index()

    stratified_g_kf = StratifiedGroupKFold(
        n_splits=int(1/test_size),
        shuffle=True,
        random_state=random_state
    )
    splits = list(stratified_g_kf.split(X=patient_df, y=patient_df['target'], groups=patient_df['patient']))
    train_idx, test_idx = splits[0]

    train_patients = patient_df.iloc[train_idx]['patient']
    test_patients= patient_df.iloc[test_idx]['patient']
    
    train_df = df[df['patient'].isin(train_patients)].reset_index(drop=True)
    test_df = df[df['patient'].isin(test_patients)].reset_index(drop=True)

    return train_df, test_df

def crop_to_brain(img):
    """
    Crop the image to the brain region.
    
    We do this since the MRI jpgs have HUGE black borders around the brain. An
    alternative would be to mask the image where values of gray ara > 0.05, But
    this would mask regions inside the brain where there are such values
    """
    mask = img != 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rowmin, rowmax = np.where(rows)[0][[0, -1]]
    colmin, colmax = np.where(cols)[0][[0, -1]]

    return img[rowmin:rowmax+1, colmin:colmax+1]

def mri_jpg_to_graph(mri_path, n_segments=30):
    """
    Create the graph for a single layer of an MRI image
    """
    img = imread(mri_path)
    img = crop_to_brain(img)
    gray = rgb2gray(img)

    segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)

    total_pixels = gray.size
    num_segs = segments.max() + 1
    total_pixels = gray.size
    mean_intensities = mean(gray, labels=segments, index=np.arange(num_segs))
    var_intensities = variance(gray, labels=segments, index=np.arange(num_segs))
    areas = np.bincount(segments.ravel()) / total_pixels
    centroids = center_of_mass(np.ones_like(gray), labels=segments, index=np.arange(num_segs))

    node_feats = torch.tensor(
        np.stack([mean_intensities, var_intensities, areas], axis=1), 
        dtype=torch.float
    )
    pos = torch.tensor(
        np.array([[c[1], c[0]] for c in centroids]), 
        dtype=torch.float
    )

    adj = set()

    # horizontal neighbors
    diff_right = segments[:,:-1] != segments[:, 1:]
    edges_right = np.stack([segments[:, :-1][diff_right], segments[:, 1:][diff_right]], axis=1)
    for edge in edges_right:
        adj.add((edge[0], edge[1]))
        adj.add((edge[1], edge[0]))

    # vertical neighbors
    diff_down = segments[:-1, :] != segments[1:, :]
    edges_down = np.stack([segments[:-1, :][diff_down], segments[1:, :][diff_down]], axis=1)
    for edge in edges_down:
        adj.add((edge[0], edge[1]))
        adj.add((edge[1], edge[0]))

    edge_index = torch.tensor(list(adj),dtype=torch.long).t().contiguous()

    return Data(x=node_feats, edge_index=edge_index, pos=pos)


# We define the following class by extending torch_geometric's Dataset
class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super(MRIDataset, self).__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = {
            'Non Demented': 0,
            'Very mild Dementia': 1,
            'Mild Dementia': 2
        }
        self.transform = transform

    def len(self):
        return len(self.dataframe)
    
    def get(self, idx):
        row = self.dataframe.iloc[idx]
        graph = mri_jpg_to_graph(row['path'])
        graph.y = torch.tensor([self.label_map[row['target']]], dtype=torch.long)
        return graph
    
class MRISeqDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, n_segments):
        self.scans = dataframe.groupby(['patient', 'mri', 'scan'])
        self.samples = list(self.scans)
        self.label_map = {
            'Non Demented': 0,
            'Very mild Dementia': 1,
            'Mild Dementia': 2,
        }
        self.n_segs = n_segments

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        _, scan = self.samples[idx]
        label = scan['target'].iloc[0]

        graphs = []
        for _, row in scan.sort_values('layer').iterrows():
            graphs.append(mri_jpg_to_graph(row['path']))

        return graphs, torch.tensor(self.label_map[label], dtype=torch.long)
    
def custom_collate(batch):
    sequences, labels = zip(*batch) # List with elements as: (List[Data], label)
    return list(sequences), torch.stack(labels)

if __name__ == "__main__":
    import time
    import sys
    directory = './Data'

    print("#####################Directory check#################################")
    filenames = get_filenames(directory)
    for category in filenames.keys():
        print(f'{category}: #filenames = {len(filenames[category])}')
        # print(filenames[category][:5])

    print("####################Regex check######################################")
    test = filenames['Non Demented'][30000:30002]
    for item in test:
        print(item)
        print(extract_MRI_info(item))

    print("#####################Df creation check###############################")
    df = create_dataframe(directory)
    print(df.head())

    print("#####################Dataset check###################################")
    dataset = MRIDataset(df)
    sample_idx=3000
    graph = dataset[sample_idx]

    # from graph_plot import visualize_data_object
    # # visualize_data_object(graph, df['path'][sample_idx])

    # from graph_plot import visualize_data_object_full
    # visualize_data_object_full(graph, df['path'][sample_idx], n_segments=30)

    # Check consistency of target labels
    print("################Label consistency check##############################")
    inconsistent = (
        df.groupby('patient')['target']
        .nunique()
        .reset_index()
        .query('target > 1')
    )

    if inconsistent.empty:
        print("We are ok! One diagnosis per patient")
    else:
        print("There are patients with multiple diagnoses")
        print(inconsistent)

    print("######################unique patients check##########################")
    print(f"len(full_df): {len(df)}")
    print(
        df.groupby('patient')['target'].first().value_counts()
    )
    start = time.time()
    train_df, test_df = stratified_patient_split(df)
    end = time.time()
    print(f"Time taken for split: {end-start} s")
    print(f"len(train_df): {len(train_df)}")
    print(
        train_df.groupby('patient')['target'].first().value_counts()
    )
    print(f"len(test_df): {len(test_df)}")
    print(
        test_df.groupby('patient')['target'].first().value_counts()
    )
    print(f"len(train)+len(test): {len(train_df)+len(test_df)}")

    print("##################SeqDataset check##############################################")
    from torch.utils.data import DataLoader
    dataframe = create_dataframe(directory)
    seq_dataset = MRISeqDataset(dataframe, 20)

    print(f"len(seq_dataset): {len(seq_dataset)}")
    loader = DataLoader(seq_dataset, 2, shuffle=True, collate_fn=custom_collate)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    print(f"Time to load first iter: {end-start} s")
    seqs, labels = batch
    print("Loaded Batch n.1:")
    print(f"Labels: {labels} Num Sequences:{len(seqs)}")
    for i, seq in enumerate(seqs):
        print(f"len of seq n.{i}: {len(seq)}")
    batch = next(iter(loader))
    seqs, labels = batch
    print("Loaded Batch n.2:")
    print(f"Labels: {labels} Num Sequences:{len(seqs)}")
    for i, seq in enumerate(seqs):
        print(f"len of seq n.{i}: {len(seq)}")