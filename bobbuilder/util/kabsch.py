import numpy as np

def kabsch_algorithm(coords1, coords2, center=True):
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix.

    Parameters:
    - coords1: Numpy array of shape (3, N) representing the first set of coordinates.
    - coords2: Numpy array of shape (3, N) representing the second set of coordinates.
    - center: Boolean, whether to center the coordinates or not. Default is True.

    Returns:
    - R: Optimal rotation matrix.
    - t: Translation vector.
    """

    if center:
        # Center the coordinates by subtracting the mean
        center1 = np.mean(coords1, axis=1, keepdims=True)
        center2 = np.mean(coords2, axis=1, keepdims=True)

        centered_coords1 = coords1 - center1
        centered_coords2 = coords2 - center2
    else:
        centered_coords1 = coords1
        centered_coords2 = coords2

    # Calculate the covariance matrix H
    H = np.dot(centered_coords1, centered_coords2.T)

    # Use Singular Value Decomposition (SVD) to factorize H
    U, _, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix R
    R = np.dot(Vt.T, U.T)

    # Calculate the translation vector t
    t = center2 - np.dot(R, center1) if center else np.zeros((3, 1))

    return R, t