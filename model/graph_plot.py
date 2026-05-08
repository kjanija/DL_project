import matplotlib.pyplot as plt
import networkx as nx
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from dataset import crop_to_brain

def visualize_data_object(data, original_img_path):
    """
    Visualize original image, superpixel segmentation, and graph overlay using a PyG Data object.
    """
    # Load image
    img = imread(original_img_path)
    img = crop_to_brain(img)

    # Convert PyG edge_index to NetworkX format
    edge_index = data.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))

    # Convert node positions from tensor to dict for networkx
    pos = {i: (data.pos[i][0].item(), data.pos[i][1].item()) for i in range(data.num_nodes)}

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Graph on image
    axs[1].imshow(img, cmap='gray')
    nx.draw(G, pos, ax=axs[1], node_size=30, edge_color='cyan', node_color='red', with_labels=False)
    axs[1].set_title("Graph Overlay")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

from skimage.io import imread
from skimage.segmentation import mark_boundaries, slic
from skimage.color import rgb2gray

def visualize_data_object_full(data, img_path, n_segments=100):
    """
    Visualize:
    1. Original image
    2. Superpixel segmentation
    3. Graph over grayscale image
    4. Graph over segmentation
    """
    # Load image and convert to grayscale
    img = imread(img_path)
    img = crop_to_brain(img)

    # Recreate superpixels
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)

    # Graph edges and node positions
    edge_index = data.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    pos_dict = {i: (data.pos[i][0].item(), data.pos[i][1].item()) for i in range(data.num_nodes)}

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(pos_dict.keys())
    G.add_edges_from(edges)

     # Create 2x2 figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Original image
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # 2. Superpixel segmentation
    axs[0, 1].imshow(mark_boundaries(img, segments))
    axs[0, 1].set_title("Superpixel Segmentation")
    axs[0, 1].axis("off")

    # 3. Graph over grayscale image
    axs[1, 0].imshow(img, cmap='gray')
    nx.draw(G, pos=pos_dict, ax=axs[1, 0], node_size=30, edge_color='cyan', node_color='red')
    axs[1, 0].set_title("Graph on Grayscale Image")
    axs[1, 0].axis("off")

    # 4. Graph over segmentation
    axs[1, 1].imshow(mark_boundaries(img, segments))
    nx.draw(G, pos=pos_dict, ax=axs[1, 1], node_size=30, edge_color='cyan', node_color='red')
    axs[1, 1].set_title("Graph on Segmentation")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()