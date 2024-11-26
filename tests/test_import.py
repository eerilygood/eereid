import unittest
from eereid.importhelper import importhelper

class TestImporthelper(unittest.TestCase):
    def test_getattr(self):
        with self.assertRaises(ImportError):
            np=importhelper('numpy', "working")
            np.random.seed()
            
    def test_call(self):
        with self.assertRaises(ImportError):
            np=importhelper('numpy', "working")
            res=np()