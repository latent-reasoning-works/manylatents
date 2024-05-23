import unittest
import numpy as np
from scipy.linalg import solve_banded
from fokker_plank_simdata import make_fp_implicit, make_fp_explicit

def simulate_wiener_process(dx, dt, sigma, time_steps, N, random_seed=42):
    np.random.seed(random_seed)
    x = np.arange(-5, 5, dx)

    # Initial condition (e.g., Delta function at zero)
    p = np.zeros_like(x)
    p[int(len(x) / 2)] = 1 / dx  # Normalize to form a probability density

    # Generate Wiener process samples
    for t in range(time_steps):
        wiener_samples = np.random.normal(loc=0.0, scale=sigma * np.sqrt(dt), size=len(x))
        p += wiener_samples
        p = np.clip(p, 0, None)  # Ensure no negative probabilities
        p /= p.sum() * dx  # Normalize to maintain total probability

    return p

class TestFokkerPlanckMethods(unittest.TestCase):
    def setUp(self):
        self.dx = 0.1
        self.dt = 0.001  # Smaller time step for explicit method stability
        self.sigma = 0.5
        self.k = 5.0
        self.time_steps = 1  # Only a few steps for testing
        self.N = 10
        self.random_seed = 42
        
    def test_implicit_fp_consistency(self):
        implicit_p1, _, _ = make_fp_implicit(self.dx, self.dt, self.sigma, self.k, 
                                       self.time_steps, self.N, self.random_seed)
        implicit_p2, _, _ = make_fp_implicit(self.dx, self.dt, self.sigma, self.k, 
                                       self.time_steps, self.N, self.random_seed)
        
        # Check if the probabilities are similar
        np.testing.assert_allclose(implicit_p1, implicit_p2, atol=1e-2)        

    def test_explicit_fp_consistency(self):
        explicit_p1, _, _ = make_fp_explicit(self.dx, self.dt, self.sigma, self.k, 
                                       self.time_steps, self.N, self.random_seed)
        explicit_p2, _, _ = make_fp_explicit(self.dx, self.dt, self.sigma, self.k, 
                                       self.time_steps, self.N, self.random_seed)
        
        # Check if the probabilities are similar
        np.testing.assert_allclose(explicit_p1, explicit_p2, atol=1e-2)  

    def test_explicit_fp_vs_implicit(self):
        
        assert self.dt <= (self.dx**2)/(2*(self.sigma**2))
        
        implicit_p, _, _ = make_fp_implicit(self.dx, self.dt, self.sigma, self.k, 
                                      self.time_steps, self.N, self.random_seed)
        explicit_p, _, _ = make_fp_explicit(self.dx, self.dt, self.sigma, self.k, 
                                      self.time_steps, self.N, self.random_seed)
        
        # Check if the probabilities are similar
        #print(implicit_p[:])
        #print(implicit_p.max(), implicit_p.argmax())
        #print((np.round(implicit_p, 2)>0).sum())
   
        #print(explicit_p[:])
        #print(explicit_p.max(), explicit_p.argmax())
        #print((np.round(explicit_p, 2)>0).sum())
        
        np.testing.assert_allclose(implicit_p, explicit_p, atol=1e-2)
        
    def test_implicit_fp_vs_wiener(self):
        k = 0.0  # No drift for Wiener process
    
        implicit_p, _, _ = make_fp_implicit(self.dx, self.dt, self.sigma, k, 
                                      self.time_steps, self.N, self.random_seed)
        wiener_p = simulate_wiener_process(self.dx, self.dt, self.sigma, 
                                              self.time_steps, self.N, self.random_seed)
        
        # Check if the probabilities are similar
        #print('Implicit (no k)')
        #print(np.round(implicit_p[:], 2))
        #print(implicit_p.max(), implicit_p.argmax())
        #print((np.round(implicit_p, 2)>0).sum())
        
        #print('Weiner')
        #print(np.round(wiener_p[:], 2))
        #print(wiener_p.max(), wiener_p.argmax())
        #print((np.round(wiener_p, 2)>0).sum())
    
        np.testing.assert_allclose(implicit_p, wiener_p, atol=1e-2)

    def test_explicit_fp_vs_wiener(self):
        k = 0.0  # No drift for Wiener process
    
        explicit_p, _, _ = make_fp_explicit(self.dx, self.dt, self.sigma, k, 
                                      self.time_steps, self.N, self.random_seed)
        wiener_p = simulate_wiener_process(self.dx, self.dt, self.sigma, self.time_steps, 
                                              self.N, self.random_seed)
        
        # Check if the probabilities are similar
        #print('Explicit (no k)')
        #print(np.round(explicit_p[:], 2))
        #print(explicit_p.max(), explicit_p.argmax())
        #print((np.round(explicit_p, 2)>0).sum())
  
        #print('Weiner')
        #print(np.round(wiener_p[:], 2))
        #print(wiener_p.max(), wiener_p.argmax())
        #print((np.round(wiener_p, 2)>0).sum())

        # Check if the probabilities are similar
        np.testing.assert_allclose(explicit_p, wiener_p, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
