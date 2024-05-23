'''
Diffusion distance is defined by:
$D(x_i,x_j)^2 = \sum_k (P_{ik} - P_{jk})^2/\pi(x_k)$

The theory of diffusion maps is that we can eigendecompose our $P$ matrix and create maps for each point $x_i$. 

Specifically, let the eigendecomposition be: $P=\Psi\Lambda\Phi$ where:
* the right eigenvectors $\Psi=[\psi_1, ..., \psi_N]$ with $\psi_l=[\psi_l(x_1),...,\psi_l(x_N)]^T$
* $\Lambda$ be diagonal matrix of eigenvalues $\lambda_l$.

The diffusion map for $x_i$ is the ith row of: $DM(x_i)=\Psi\Lambda$

Eucidean distances of these maps preserve the "diffusion distances" of the points in ambient space:

$D(x_i,x_j)^2 = \sum_l \lambda_l^2 (\psi_l(x_i)-\psi_l(x_j))^2 = ||DM(x_i)-DM(x_j)||^2$

Implementation detail:

We eigendecompose the symmetric conjugate of $P$. Note that naive eigendecomposition of $P$ may result in non-orthogonal eigenvectors.
'''

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.spatial.distance import euclidean
import sklearn
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns


def compute_dm(K, alpha=0.):
    # Using setup and notation from: https://www.stats.ox.ac.uk/~cucuring/CDT_08_Nonlinear_Dim_Red_b__Diffusion_Maps_FoDS.pdf
    # alpha=0 Graph Laplacian
    # alpha=0.5 Fokker-Plank operator
    # alpha=1 Laplace-Beltrami operator

    d_noalpha = K.sum(1).flatten()

    # weighted graph Laplacian normalization
    d = d_noalpha**alpha 
    D = np.diag(d)
    D_inv = np.diag(1/d)

    # Normalize K according to α
    K_alpha = D_inv@K@D_inv

    d_alpha = K_alpha.sum(1).flatten()
    D_alpha = np.diag(d_alpha)
    D_inv_alpha = np.diag(1/d_alpha)

    L = D_inv_alpha@K_alpha # anisotropic transition kernel (AKA diffusion operator)

    # build symmetric matrix
    D_sqrt_inv_alpha = np.diag(1/np.sqrt(d_alpha))
    D_sqrt_alpha = np.diag(np.sqrt(d_alpha))
    S = D_sqrt_inv_alpha@K_alpha@D_sqrt_inv_alpha

    # spectral decomposition of S
    # IMPORTANT NOTE:
    # In theory you could run np.linalg.eig(L),
    # BUT this returns non-orthogonal eigenvectors!
    # So would have to correct for that
    # Using SVD since more numerically stable
    evecs, svals, _ = scipy.linalg.svd(S)

    # Retrieve sign!
    test_product = S@evecs
    expected_product = evecs@np.diag(svals)
    signs = np.isclose(expected_product, test_product).mean(0)
    signs[~np.isclose(signs, 1)] = -1
    evals = svals*signs

    # convert right eigenvectors of S to those of L
    evecs_right = D_sqrt_inv_alpha@evecs
    evecs_left = D_sqrt_alpha@evecs

    # make sure ordered by eigenvalue
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    #evecs = evecs[:, order]
    evecs_left = evecs_left[:, order]
    evecs_right = evecs_right[:, order]

    # Scaling factor for eigenvectors
    #scaling_factor = 1/d_noalpha.sum()
    #scaling_factor = evecs_right[0, 0]
    scaling_factor = 1/np.sqrt(d_alpha.sum())
    if np.isclose(evecs_right[:,0]/scaling_factor, -1).any():
        scaling_factor *= -1

    # Apply scaling
    evecs_right /= scaling_factor

    # Adjust left eigenvectors to maintain eigendecomposition
    # Assuming evecs_right[0,0] is non-zero for all practical cases
    evecs_left *= scaling_factor

    # Safety Checks!
    # First left eigenvector is stationary distribution
    neg_evals = evals < 0
    if neg_evals.sum() > 0:
        print("{} eigenvalues are negative: min={}".format(len(evals[neg_evals]),
                                                           evals[neg_evals].min()))
    one_evals = np.isclose(evals, 1).sum()
    if one_evals > 1:
        print("{} eigenvalues are 1".format(one_evals))
    if not np.allclose(evecs_left[:,0]/evecs_left.sum(), d_alpha/d_alpha.sum()):
        print("left evec not exactly stationary dist. Proceed with caution!")
    # First right eigenvector is all 1s
    if not  np.allclose(evecs_right[:,0], 1):
        print("right evec not trivial (1s)! Proceed with caution!")
    # Decomposition is correct
    if not np.allclose(L, evecs_right@np.diag(evals)@evecs_left.T):
        print("evals/evecs dont exactly match with diffusion operator. Proceed with caution!")


    #diffusion_coords = evecs_right@np.diag(evals)
    
    # return eigenvectors and eigenvalues instead so that we can compute DMs for different t's
    return evecs_right, evals, L, d_noalpha


## Helper Functions
def compute_diffusion_distance_direct(P: np.ndarray, pi: np.ndarray, i: int) -> np.ndarray:
    """
    Compute the squared diffusion distance from point i to all other points using the diffusion matrix P.

    Parameters:
    P (np.ndarray): NxN diffusion matrix where P[i, j] is the transition probability from i to j.
    pi (np.ndarray): Stationary distribution, 1D array of length N.
    i (int): Index of the point from which distances are calculated.

    Returns:
    np.ndarray: A 1D array of diffusion distances from point i to all other points.
    """
    return np.array([euclidean(P[i], P[j], w=1/pi) for j in range(len(P))])**2

def compute_diffusion_distance_using_dmcoords(dm_coords, i):
    return pairwise_distances(dm_coords)[i]

def plot(data, palette, labels, label_order, ax=None, label_positions=True, **kwargs):
    if not ax:
        create_new_plot = True
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        create_new_plot = False

    sns.scatterplot(x=data[:, 0], 
                    y=data[:, 1], 
                    hue=labels, 
                    palette=palette, 
                    ax=ax, 
                    hue_order=label_order,
                    **kwargs)

    if create_new_plot:
        #ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    else:
        # we dont need ticks or ticklabels
        ax.get_legend().remove()
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())

    if label_positions:
        for label in label_order:
            indices = np.where(np.array(labels) == label)
            mean_position = np.mean(data[indices], axis=0)
            ax.text(mean_position[0], mean_position[1], label, weight='bold', ha='center')

def plot_change_in_diffusion_distance(P, pi, diffusion_coords, dims_to_plot):
    Dt = [compute_diffusion_distance_direct(P, pi, i) for i in range(len(P))]
    Dt = np.vstack(Dt) # ground truth

    distortion = []
    assert dims_to_plot[0] == 1
    assert dims_to_plot[-1] == len(diffusion_coords)

    for dims_keep in dims_to_plot:
        diff_dists = sklearn.metrics.pairwise_distances(diffusion_coords[:,:dims_keep])**2
        distortion.append(np.sqrt(((diff_dists - Dt)**2).mean()))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=dims_to_plot, y=distortion)
    ax.set_xlabel('# Dims to Keep')
    ax.set_ylabel('RMSE diffusion dist vs Euclid. dist. of DM embs')

def display_powers_of_diff_op(P, t_to_display, percentile_thresh=99.5):
    evals, evecs = np.linalg.eig(P)
    evecs_inv = np.linalg.inv(evecs)
    fig, ax = plt.subplots(ncols=5, figsize=(50,10))
    for i,t in enumerate(t_to_display):
        Pt = np.array(evecs@np.diag(evals**t)@evecs_inv)
        ax[i].imshow(Pt, vmin=0, 
                     vmax=np.percentile(Pt, percentile_thresh), 
                     cmap='Reds', 
                     aspect='auto')
