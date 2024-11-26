import unittest
import eereid as ee

class NoveltyDistanceTest(unittest.TestCase):
    def test_manhatten(self):
        g=ee.ghost(dataset=ee.datasets.mnist(),
                distance=ee.distances.lN(2),
                loss=ee.losses.extended_triplet(),
                model=ee.models.conv(),
                preprocessing=ee.prepros.subsample(0.1),
                novelty=ee.novelty.distance(ee.distances.manhattan()),
                triplet_count=100)
            
        g["epochs"] = 1
        g.evaluate()

