import unittest
import numpy as np
from auto_label import select_labels
class MaskSelectionTests(unittest.TestCase):
    def test_low_score_ignored(self):
        self.assertEqual(int(select_labels(np.full((4,1,1),0.3))[0,0]),255)
    def test_ambiguous_ignored(self):
        self.assertEqual(int(select_labels(np.array([.7,.69,.2,.1])[:,None,None])[0,0]),255)
    def test_winner_keeps_class_id(self):
        self.assertEqual(int(select_labels(np.array([.1,.2,.8,.1])[:,None,None])[0,0]),2)
if __name__=='__main__':unittest.main()
