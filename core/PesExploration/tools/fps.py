import numpy as np
from itertools import combinations
import yaml


def generate_n_compositions(elements_boundary, total_atoms=48, min_others=9):
    """Generate all valid N-element compositions for a given atom count.

    Uses the stars-and-bars method to enumerate all non-negative integer
    partitions of *total_atoms* into len(elements_boundary) parts, then
    filters by per-element min/max bounds.

    Args:
        elements_boundary: dict mapping element symbol to [min, max] count.
        total_atoms: total number of atoms in the composition.
        min_others: minimum count of non-primary elements (legacy constraint).

    Returns:
        np.ndarray of shape (n_compositions, n_elements).
    """

    # Use stars-and-bars method to generate all non-negative integer sequences summing to total_atoms
    # Logic: choose (n - 1) divider positions out of (total + n - 1) positions
    def compositions_generator(n, total):
        if n == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions_generator(n - 1, total - i):
                yield (i,) + rest

    all_comps = []
    for comp in compositions_generator(len(elements_boundary.keys()), total_atoms):
        # Apply constraint: no element count may exceed its bounds
        keys = elements_boundary.keys()
        if all(elements_boundary[key][0] <= x <= elements_boundary[key][1] for key, x in zip(keys, comp)):
            all_comps.append(comp)

    return np.array(all_comps)


def farthest_point_sampling(points, k):
    """Farthest-point sampling in N-dimensional space.

    Args:
        points: array of shape (n_samples, n_features).
        k: number of points to select.

    Returns:
        np.ndarray of selected indices.
    """
    if len(points) <= k:
        return np.arange(len(points))

    selected = []
    # 1. Randomly select a starting point
    idx = np.random.randint(len(points))
    selected.append(idx)

    # Initialize distance array to infinity
    distances = np.full(len(points), np.inf)

    for _ in range(k - 1):
        last_point = points[selected[-1]]
        # Compute Euclidean distance from current point to all points (N-dimensional)
        dist = np.linalg.norm(points - last_point, axis=1)
        # Update minimum distance from each point to the selected set
        distances = np.minimum(distances, dist)
        # Select the point farthest from the selected set
        next_idx = np.argmax(distances)
        selected.append(next_idx)

    return np.array(selected)


def sample_compositions_n(elements_boundary, total_atoms, substrate, n_samples):
    """Sample diverse compositions via farthest-point sampling.

    Generates all valid compositions, normalizes to ratio space, applies FPS
    to select a diverse subset, then adds substrate atom counts back.

    Args:
        elements_boundary: dict mapping element symbol to [min, max] count.
        total_atoms: total number of atoms (excluding substrate).
        substrate: dict mapping element symbol to fixed substrate atom count.
        n_samples: number of compositions to select.

    Returns:
        List of composition lists, each with substrate counts added.
    """
    # Step 1: Generate all valid compositions in N-dimensional space
    all_comps = generate_n_compositions(elements_boundary, total_atoms)

    # Step 2: Normalize to ratio space (0~1) for FPS computation
    ratios = all_comps / total_atoms

    # Step 3: Perform FPS sampling
    idx = farthest_point_sampling(ratios, n_samples)

    update_compos = []
    for compos in all_comps[idx]:
        _compos = list(compos)
        for i, elem in enumerate(elements_boundary.keys()):
            _compos[i] += substrate.get(elem, 0)
        update_compos.append(_compos)
            # print(compos, _compos)

    return update_compos

def sample_composition(config_path):
    """Sample compositions from a YAML config file.

    Args:
        config_path: path to YAML file with keys elements_boundary,
            total_atoms, substrate, and n_samples.
    """
    config = yaml.safe_load(open(config_path))
    result = sample_compositions_n(elements_boundary=config["elements_boundary"],
                                   total_atoms=config["total_atoms"],
                                   substrate=config["substrate"],
                                   n_samples=config["n_samples"])
    print("All compositions:")
    print(result)


if __name__ == '__main__':
    sample_composition(config_path='<YOUR_FPS_CONFIG_PATH>')



