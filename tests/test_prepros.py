import unittest
import eereid as ee


class CropTest(unittest.TestCase):
    def test_crop(self):
        g=ee.ghost(dataset=ee.datasets.mnist(),
                distance=ee.distances.lN(2),
                loss=ee.losses.extended_triplet(),
                model=ee.models.conv(),
                prepros=[ee.prepros.crop(5,5,5,5)],
                preprocessing=ee.prepros.subsample(0.1),
                triplet_count=100)

        g["triplet_count"]=1000
        g["epochs"] = 1
        g.evaluate()