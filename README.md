# Classification of Alzheimer's Disease using GNNs and LSTMs

## Project Overview
This repository contains a deep learning pipeline designed to classify the severity of Alzheimer's disease from MRI brain scans.Using the OASIS-1 dataset, the model categorizes scans into three progressive stages: Non-Demented, Very Mild Dementia, and Mild/Moderate Dementia. 

Because Alzheimer's is a progressive disease and brain structures are highly interconnected, this project moves away from standard Convolutional Neural Networks (CNNs). Instead, it represents the brain as a graph to capture complex spatial relationships and utilizes recurrent neural networks to capture the volumetric progression across sequential 2D MRI slices.

## Methodology
The pipeline consists of three core phases:

* **Data Preprocessing & Graph Construction:** 
  * Individual MRI slices are cropped to the brain region to remove empty background space. 
  * Slices are segmented into superpixels using the SLIC algorithm.   
  * A spatial graph is constructed where each superpixel is a node, and edges are drawn between adjacent superpixels. Node features include mean intensity, variance, and area.
* **Spatial Feature Extraction (GAT):** 
  * A Graph Attention Network (GAT) processes each slice's graph.   
  * The GAT learns the importance of neighboring superpixels generating a rich embedding for each individual 2D slice.
* **Sequential Volumetric Modeling (LSTM):** 
  * An MRI scan consists of multiple sequential slices (~61 per scan).
  * An LSTM network treats the sequence of GAT embeddings as a time-series/sequential input. 
  * This allows the model to process the full 3D context of the brain volume, ultimately outputting the final severity classification.

## Repository Structure

* `dataset.py`: Contains the data loading and preprocessing logic. Handles regex-based metadata extraction from filenames, superpixel graph generation, PyTorch Geometric dataset classes (`MRIDataset`, `MRISeqDataset`), and a stratified patient-wise split to prevent data leakage.
* `GAT_model.py` / `model.py`: Defines the Graph Attention Network architectures (`GATModel`, `GATClassifier`) using PyTorch Geometric, including global mean pooling for graph-level embeddings.
* `LSTM_model.py`: Defines the core `LSTMModel` which wraps the GAT encoder and passes its outputs into a Bidirectional LSTM, followed by dimensionality reduction and the final classification head.
* `graph_plot.py`: Visualization utilities using `networkx` and `matplotlib` to plot superpixel boundaries and overlay the generated graphs directly onto the MRI images.
* `final.ipynb` / `model.ipynb`: Jupyter notebooks containing the training and validation loops, hyperparameter configurations, and the implementation of a `WeightedRandomSampler` to handle the heavily unbalanced dataset.

## Handling Class Imbalance
The dataset exhibits significant class imbalance. To mitigate this without naively duplicating image files, the training loop implements a `WeightedRandomSampler`. Additionally, cross-entropy loss weights are calculated dynamically based on the inverse frequency of each class in the training set.

## Setup and Usage

**Prerequisites:**
Ensure you have Python 3.x installed along with the following primary dependencies:
* `torch`
* `torch_geometric`
* `scikit-image`
* `scikit-learn`
* `pandas`, `numpy`, `matplotlib`, `networkx`

**Data Preparation:**
1. Download the OASIS-1 MRI JPG dataset.
2. Place the categorized folders (e.g., 'Non Demented', 'Very mild Dementia', etc.) inside a `Data/` directory at the root of the project.

**Training:**
Run the training loops provided in `final.ipynb` or execute the testing routines in the scripts to initialize the GAT-LSTM pipeline.
