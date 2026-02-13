"""
Similarity space utilities for MET-based allocation methods.

This module provides shared functionality for calculating similarity between worldviews
and embedding them in 2D space for credence-based and score-based MET methods.

"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import MDS


def calculate_pairwise_similarities(worldviews, projects):
    """
    Calculate pairwise Pearson and rank correlation matrices between worldviews.
    
    Args:
        worldviews: List of Worldview objects
        projects: List of Project objects
    
    Returns:
        (pearson_matrix, rank_matrix): Two n×n numpy arrays where [i,j] is the
                                       similarity between worldview i and worldview j
    """
    n_worldviews = len(worldviews)
    pearson_matrix = np.zeros((n_worldviews, n_worldviews))
    rank_matrix = np.zeros((n_worldviews, n_worldviews))
    
    for i, worldview_i in enumerate(worldviews):
        for j, worldview_j in enumerate(worldviews):
            if i == j:
                # Perfect self-similarity
                pearson_matrix[i, j] = 1.0
                rank_matrix[i, j] = 1.0
            else:
                # Get values for all projects from both worldviews
                values_i = []
                values_j = []
                
                for project in projects:
                    # Get evaluation from each worldview
                    # Adjust method name based on actual Worldview API
                    val_i = worldview_i.evaluate(project)
                    val_j = worldview_j.evaluate(project)
                    values_i.append(val_i)
                    values_j.append(val_j)
                
                # Calculate Pearson correlation (linear relationship of cardinal values)
                pearson_corr, _ = pearsonr(values_i, values_j)
                # Normalize to [0, 1] range
                pearson_matrix[i, j] = (pearson_corr + 1) / 2
                
                # Calculate rank correlation (agreement on ordinal rankings)
                rank_corr, _ = spearmanr(values_i, values_j)
                # Normalize to [0, 1] range
                rank_matrix[i, j] = (rank_corr + 1) / 2
    
    return pearson_matrix, rank_matrix


def embed_worldviews_in_2d_space(pearson_matrix, rank_matrix):
    """
    Embed worldviews in 2D similarity space using MDS.
    
    Creates a 2D coordinate system where:
    - x-axis represents Pearson correlation dimension
    - y-axis represents rank correlation dimension
    - Distance between points indicates dissimilarity
    
    Args:
        pearson_matrix: n×n array of pairwise Pearson similarities
        rank_matrix: n×n array of pairwise rank similarities
    
    Returns:
        positions: n×2 array where positions[i] = [x, y] coordinates of worldview i
    """
    n_worldviews = pearson_matrix.shape[0]
    
    # Combine both similarity measures into a distance matrix
    # Distance = sqrt((1-pearson)² + (1-rank)²)
    distance_matrix = np.zeros((n_worldviews, n_worldviews))
    
    for i in range(n_worldviews):
        for j in range(n_worldviews):
            pearson_dist = 1 - pearson_matrix[i, j]
            rank_dist = 1 - rank_matrix[i, j]
            distance_matrix[i, j] = np.sqrt(pearson_dist**2 + rank_dist**2)
    
    # Use multidimensional scaling to find 2D embedding
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)
    
    return positions


def calculate_weighted_centroid(positions, weights):
    """
    Calculate weighted centroid (center of mass) in 2D space.
    
    Args:
        positions: n×2 array of positions
        weights: n-length array of weights for each position
    
    Returns:
        centroid: [x, y] coordinates of weighted centroid
    """
    if np.sum(weights) == 0:
        return np.array([0.0, 0.0])
    
    centroid = np.average(positions, axis=0, weights=weights)
    return centroid


def find_closest_worldview(worldview_positions, target_point):
    """
    Find the worldview whose position is closest to the target point.
    
    Args:
        worldview_positions: n×2 array of worldview positions
        target_point: [x, y] coordinates to find nearest worldview to
    
    Returns:
        Index of closest worldview
    """
    from scipy.spatial.distance import euclidean
    
    distances = [euclidean(pos, target_point) for pos in worldview_positions]
    return int(np.argmin(distances))
