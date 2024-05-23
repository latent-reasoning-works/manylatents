import unittest
import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import adjusted_rand_score
import graphtools
import phate
from diffusion_maps import compute_dm, compute_diffusion_distance_direct, compute_diffusion_distance_using_dmcoords 
from graph_utils import get_groups_from_graph

def get_groups_from_eigenvectors(evecs_right):
    # get trivial eigenvectors and which samples contain each
    trivial_vecs_groups = np.zeros(len(evecs_right))
    for i in range(len(evecs_right)):
        zero_idx = np.isclose(evecs_right[:,i],0)
        if zero_idx.sum() > 0:
            max_val = evecs_right[:,i][~zero_idx].min()
            min_val = evecs_right[:,i][~zero_idx].max()
            if np.isclose(max_val, 
                          min_val):
                trivial_vecs_groups[~zero_idx] = i
    return trivial_vecs_groups

class TestDiffusionMaps(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_dm_consistency_sim_data(self):
        #data, labels = phate.tree.gen_dla() # tree data
        X = np.random.rand(20, 10)
        K = pairwise_kernels(X, metric='rbf', gamma=2)
        evecs_right, evals, _, _ = compute_dm(K)
        diffusion_coords = evecs_right@np.diag(evals)
        
        evecs_right2, evals2, _, _ = compute_dm(K)
        diffusion_coords2 = evecs_right2@np.diag(evals2)

        np.testing.assert_allclose(diffusion_coords, 
                                   diffusion_coords2, atol=1e-2)  

    def test_dm_dist_exact_tree_data(self):
        #data, labels = phate.tree.gen_dla() # tree data
        np.random.seed(42)
        data, labels = phate.tree.gen_dla(n_branch=7) # tree data
        K = pairwise_kernels(data, metric='rbf', gamma=1/1000)

        evecs_right, evals, P, d = compute_dm(K)
        diffusion_coords = evecs_right@np.diag(evals)

        test_point = 4
        Dt = compute_diffusion_distance_direct(P, d/d.sum(), test_point)
        Dt2 = compute_diffusion_distance_using_dmcoords(diffusion_coords,
                                                        test_point)**2

        np.testing.assert_allclose(Dt, Dt2, atol=1e-2)
        
    def test_dm_dist_exact_disconnected_digits_data(self):
        digits = datasets.load_digits()
        data = digits['data']
        targets = digits['target']

        G = graphtools.Graph(data, 
                             decay=60, # make sure disconnected
                             n_landmark=None)
        K = G.kernel
        K = np.array(K.todense())
        #P = G.diff_op

        evecs_right, evals, P, d = compute_dm(K)
        diffusion_coords = evecs_right@np.diag(evals)

        test_point = 4
        Dt = compute_diffusion_distance_direct(P, d/d.sum(), test_point)
        Dt2 = compute_diffusion_distance_using_dmcoords(diffusion_coords,
                                                        test_point)**2

        np.testing.assert_allclose(Dt, Dt2, atol=1e-2)    
        
        # check that trivial eigenvectors correspond to disconnected parts
        # Get groups from graph
        graph_indices = get_groups_from_graph(K)
        # get groups from eigenvectors (the trivial eigvecs)
        eig_indices = get_groups_from_eigenvectors(evecs_right)
        assert len(np.unique(graph_indices)) > 1
        assert len(np.unique(eig_indices)) > 1

        assert np.isclose(adjusted_rand_score(graph_indices,
                                              eig_indices), 1)


if __name__ == '__main__':
    unittest.main()
