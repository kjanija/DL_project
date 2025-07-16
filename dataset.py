import os
import re
import pandas as pd
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
from sympy import comp
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

    re_pattern = re.compile(r'OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+).jpg')
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

def mri_jpg_to_graph(mri_path, n_segments=100):
    """
    Create the graph for a single layer of an MRI image
    """
    img = imread(mri_path)
    img = crop_to_brain(img)
    gray = rgb2gray(img)

    segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)

    node_features = []
    positions = []

    # We get the features (mean intensity for now, we could do more later) for
    # each segment. Note that we save also the coordinate of the centroid of such segment
    for segment in np.unique(segments):
        mask = segments == segment                              # we are interested only in the current segment
        mean_intensity = np.mean(gray[mask])                    # get the mean in the current segment
        yx_coords = np.argwhere(mask)
        centroid = np.mean(yx_coords, axis=0) # [y, x]          # calc the centroid of the segment
        node_features.append([mean_intensity])  
        positions.append([centroid[1], centroid[0]]) # [x, y]

    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)

    # Now let's create the edges between segments
    heigth, width = segments.shape
    edges = set()

    # We'll be adding them by starting from the top left and proceeding to bottom-right
    for y in range(heigth-1):
        for x in range(width-1):
            curr = segments[y, x]
            right = segments[y, x+1]
            down = segments[y+1, x]

            if curr != right:
                edges.add((curr, right))
                edges.add((right, curr))
            if curr != down:
                edges.add((curr, down))
                edges.add((down, curr))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, pos=pos)


# We define the following class by extending torch_geometric's Dataset
class MRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super(MRIDataset, self).__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = {
            'Non Demented': 0,
            'Very mild Dementia': 1,
            'Mild Dementia': 2,
            'Moderate Dementia': 3
        }
        self.transform = transform

    def len(self):
        return len(self.dataframe)
    
    def get(self, idx):
        row = self.dataframe.iloc[idx]
        graph = mri_jpg_to_graph(row['path'])
        graph.y = torch.tensor([self.label_map[row['target']]], dtype=torch.long)
        return graph

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

    print("################################################################")
    dataset = MRIDataset(df)
    sample_idx=3000
    graph = dataset[sample_idx]
    print(graph)

    from graph_plot import visualize_data_object
    # visualize_data_object(graph, df['path'][sample_idx])

    from graph_plot import visualize_data_object_full
    visualize_data_object_full(graph, df['path'][sample_idx], n_segments=100)