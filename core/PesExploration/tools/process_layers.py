# Cluster substrate atoms into layers along the z-axis.
# Atoms are grouped by z-coordinate using DBSCAN (or KMeans as fallback).
from collections import Counter

from numpy import where
from sklearn.cluster import DBSCAN
from ase.io import read, write
from ase.geometry import get_distances
import os
import numpy as np
from sklearn.cluster import KMeans


def show_clusters(atoms, clusters_dict, output_dir):
    """Write each cluster of atoms to a separate CIF file.

    Args:
        atoms: full ASE Atoms object.
        clusters_dict: dict mapping cluster label to atom index list.
        output_dir: directory for output CIF files.
    """
    for key, value in clusters_dict.items():
        if key != -1:
            layer_atoms = atoms[value]
            if len(layer_atoms) == 0:
                continue
            write(os.path.join(output_dir, f"layer_{key}.cif"), layer_atoms)
    return


def process_layers(atoms,
                   layer_num=5,
                   visualize=False,
                   output_dir=None,
                   substrate_path=None):
    """Partition atoms into layers along the z-axis.

    Clusters atoms by scaled z-coordinate using DBSCAN (falls back to KMeans
    if too few clusters are found), then merges clusters into *layer_num*
    groups. Substrate atoms (identified from *substrate_path*) are separated
    into their own group.

    Args:
        atoms: ASE Atoms object.
        layer_num: desired number of non-substrate layers.
        visualize: if True, write each layer to a CIF file.
        output_dir: directory for visualization output.
        substrate_path: path to a reference structure for identifying
            substrate atoms by element and count.

    Returns:
        Dict mapping layer index to list of atom indices.
    """
    # Get atomic positions
    positions = atoms.get_scaled_positions()
    eps = 1.0 / atoms.get_cell()[2, 2]
    min_samples = 2
    # Remove substrate atoms
    surf_idx = []
    if substrate_path:
        substrate = read(substrate_path)
        elem_mun = Counter(substrate.get_chemical_symbols())
        for elem in elem_mun.keys():
            elem_idx = [(i,positions[i][2]) for i, j in enumerate(atoms.get_chemical_symbols()) if j == elem]
            elem_idx = sorted(elem_idx, key=lambda x: x[1])
            elem_idx_surface = elem_idx[elem_mun[elem]:]
            surf_idx.extend([i[0] for i in elem_idx_surface])

    # Reshape data
    z_positions = list(positions[:, 2])
    for i in surf_idx:
        z_positions[i] = 0.9

    z_positions = np.array(z_positions).reshape(-1, 1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_cluster = dbscan.fit_predict(z_positions)
    clusters_id = set(labels_cluster)
    if len(clusters_id) < layer_num + 1:
        kmeans = KMeans(n_clusters=layer_num * 2)
        labels_cluster = kmeans.fit_predict(z_positions)
        clusters_id = set(labels_cluster)
    clusters_dict = {}
    for i in clusters_id:
        clusters_dict[i] = list(where(labels_cluster == i)[0])
    noise = clusters_dict.get(-1, None)
    positions_reset = positions.copy()
    positions_reset[:, :2] = 0
    if noise:
        for n in noise:
            distances = get_distances(positions_reset[n],
                                      positions_reset,
                                      cell=atoms.get_cell(),
                                      pbc=atoms.get_pbc())[1].reshape(-1)
            sorted_distances = np.argsort(distances)
            nearst_id = None
            for i in sorted_distances:
                if i not in noise:
                    nearst_id = i
                    break
            for key, value in clusters_dict.items():
                if nearst_id in value:
                    clusters_dict[key].append(n)
                    break

    key_mean = []
    for key, value in clusters_dict.items():
        if key != -1 and value[0] not in surf_idx:
            key_mean.append((key, np.mean(positions[value][:, 2])))

    key_mean = sorted(key_mean, key=lambda x: x[1])
    key_sorted = np.array([i[0] for i in key_mean])
    key_sorted_layers = np.array_split(key_sorted, layer_num)

    new_clusters_dict = {}
    for i, key in enumerate(key_sorted_layers):
        new_clusters_dict[i] = []
        for k in key:
            new_clusters_dict[i].extend(clusters_dict[k])
    surf_clus = max([i for i in new_clusters_dict.keys()]) + 1
    new_clusters_dict[surf_clus] = surf_idx

    if visualize:
        show_clusters(atoms, new_clusters_dict, output_dir)
    return new_clusters_dict


if __name__ == '__main__':
    # example
    test = read("<YOUR_CONTCAR_PATH>")
    process_layers(test,
                   layer_num=4,
                   visualize=True,
                   output_dir="<YOUR_OUTPUT_DIR>",
                   substrate_path="<YOUR_SUBSTRATE_PATH>")
