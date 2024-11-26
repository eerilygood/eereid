import unittest
import eereid as ee
import os


class SavingTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.g=ee.ghost(dataset=ee.datasets.mnist(),
           distance=ee.distances.lN(2),
           loss=ee.losses.extended_triplet(),
           model=ee.models.conv(),
           preprocessing=ee.prepros.subsample(0.1),
           novelty=ee.novelty.knn(),
           triplet_count=100)
        
        cls.g["epochs"] = 1
        cls.g.evaluate()
        
    @classmethod
    def tearDownClass(cls):
        os.remove("testmodel.keras")
        os.remove("testdata.npz")
        return super().tearDownClass()
        
    def test_model_save(self):
        self.g.save_model("testmodel.keras")
    
    def test_data_save(self):
        self.g.save_data("testdata.npz")

