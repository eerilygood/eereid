import unittest
import numpy as np
from eereid.distances import *

class CustomTestCase(unittest.TestCase):
    def assertListAlmostEqual(self, a, b):
        for _a, _b in zip(a, b):
            self.assertNearEqual(_a, _b)
            
    def assertNearEqual(self, a, b):
        self.assertAlmostEqual(a, b, places=6, 
            msg=f"Expected {b}, but got {a}")

class TestCosineSimilarity(CustomTestCase):
    def test_distance(self):
        sim = cosine_similarity()
        
        self.assertNearEqual(sim.distance([1,1,1], [1,1,1]), 1.)
        self.assertNearEqual(sim.distance([1,1,1], [-1,-1,-1]), -1.)
        self.assertNearEqual(sim.distance([1,0,0], [0,1,0]), 0.)
        
    def test_multi_distance(self):
        sim = cosine_similarity()
            
        self.assertListAlmostEqual(sim.multi_distance([[1,1,1], [2,2,2], [3,3,3]], [1, 1, 1]), [1., 1., 1.])
        self.assertListAlmostEqual(sim.multi_distance([[1,1,1], [2,2,2], [3,3,3]], [-1, -1, -1]), [-1., -1., -1.])
        self.assertListAlmostEqual(sim.multi_distance([[1,0,1], [2,0,3], [3,0,1]], [0, 1, 0]), [0., 0., 0.])
        
        
class TestEuclideanDistance(CustomTestCase):
    def test_distance(self):
        sim = euclidean()
        
        inp_a = np.array([1, 1, 1])
        inp_b = np.array([1, 1, 1])
        expected_result = 0.
        self.assertNearEqual(sim.distance(inp_a, inp_b), expected_result)
        
        inp_a = np.array([1, 1, 1])
        inp_b = np.array([-1, -1, -1])
        expected_result = np.sqrt(12)
        self.assertNearEqual(sim.distance(inp_a, inp_b), expected_result)
        
        inp_a = np.array([1, 0, 0])
        inp_b = np.array([0, 1, 0])
        expected_result = np.sqrt(2)
        self.assertNearEqual(sim.distance(inp_a, inp_b), expected_result)
        
    def test_multi_distance(self):
        sim = euclidean()
        
        inp_a = np.array([[1,1,1], [2,2,2], [3,3,3]])
        inp_b = np.array([1, 1, 1])
        expected_result = [0., np.sqrt(3), np.sqrt(12)]
        self.assertListAlmostEqual(sim.multi_distance(inp_a, inp_b), expected_result)
        
        inp_a = np.array([[1,1,1], [2,2,2], [3,3,3]])
        inp_b = np.array([-1, -1, -1])
        expected_result = [np.sqrt(12), np.sqrt(27), np.sqrt(48)]
        self.assertListAlmostEqual(sim.multi_distance(inp_a, inp_b), expected_result)
        
        inp_a = np.array([[1,0,1], [2,0,3], [3,0,1]])
        inp_b = np.array([0, 1, 0])
        expected_result = [np.sqrt(3), np.sqrt(14), np.sqrt(11)]
        self.assertListAlmostEqual(sim.multi_distance(inp_a, inp_b), expected_result)
        
        
class TestlNDistance(CustomTestCase):
    def test_distance(self):
        dist = lN(2)
        comp = euclidean()
        
        inp_a = np.random.random((3,))
        inp_b = np.random.random((3,))
        self.assertEqual(dist.distance(inp_a, inp_b), comp.distance(inp_a, inp_b))
        
        dist = lN(-1)
        inp_a = np.array([1, 1, 1])
        inp_b = np.array([0, 0, 0])
        self.assertNearEqual(dist.distance(inp_a, inp_b), 1/3)
        
        
    
    def test_multi_distance(self):
        dist = lN(2)
        comp = euclidean()
        
        inp_a = np.random.random((3,3))
        inp_b = np.random.random((3,))
        self.assertEqual(dist.distance(inp_a, inp_b), comp.distance(inp_a, inp_b))
        
        dist = lN(-1)
        inp_a = np.array([[1, 3, 4], [2, 2, 2], [3, 3, 3]])
        inp_b = np.array([0, 1, 1])
        expected_result = [0.54545454545454, 0.4, 0.75]
        self.assertListAlmostEqual(dist.multi_distance(inp_a, inp_b), expected_result)
        
    
# class TestMahalanobis(CustomTestCase):

#     def test_multi_distance(self):
#         mahalanobis = mahalanobis()
        
#         A = np.array([[1.2, 0.7, 3.2], [2.5, 3.2, 4.7], [3.7, 4.2, 5.9]])
#         b = np.array([[0.2, 0.9, 1.7], [6.2, 2.7, 1.6], [8.3, 9.7, 5.6]])
#         expected_result = self._calculate_mahalanobis(A, b)
#         result = mahalanobis.multi_distance(A, b)
#         np.testing.assert_almost_equal(result, expected_result, decimal=6, 
#             err_msg=f"Expected {expected_result}, but got {result}")

#     def _calculate_mahalanobis(self, A, b):
#         """Helper function to calculate the Mahalanobis distance manually for testing."""
#         cov = np.cov(A.T)
#         inv_cov = np.linalg.inv(cov)
#         return np.sqrt(np.diag(np.dot(np.dot((A - b), inv_cov), (A - b).T)))
    
class TestManhattan(CustomTestCase):
    def test_distance(self):
        dist = manhattan()
        
        a = np.array([1, 2, 3])
        b = np.array([1, 1, 1])
        expected_result = 3.
        result = dist.distance(a, b)
        self.assertNearEqual(result, expected_result)

    def test_multi_distance(self):
        dist = manhattan()
        
        A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        b = np.array([1, 1, 1])
        expected_result = [3., 6., 9.]
        result = dist.multi_distance(A, b)
        self.assertListAlmostEqual(result, expected_result)
    
class TestChiSquare(CustomTestCase):
    def test_distance(self):
        chi = chisquare()
        
        a = np.array([1, 2, 3])
        b = np.array([1, 1, 1])
        expected_result = 1 + 1/3
        result = chi.distance(a, b)
        self.assertNearEqual(result, expected_result)

    def test_multi_distance(self):
        chi = chisquare()
        
        A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        b = np.array([1, 1, 1])
        expected_result = [1 + 1/3, 3 + 2/15, 5.4666666666666]
        result = chi.multi_distance(A, b)
        self.assertListAlmostEqual(result, expected_result)
