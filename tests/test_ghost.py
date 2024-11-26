import unittest
import eereid as ee

class GhostTest(unittest.TestCase):
    def test_evaluate(self):
        g=ee.ghost(model=ee.models.conv(),
           dataset=ee.datasets.mnist(),
           preprocessing=ee.prepros.subsample(0.1),
           prepros=ee.prepros.apply_func())
        
        #just to speed up the evaluation
        g["triplet_count"]=1000
        g["epochs"] = 1
        
        g.evaluate()