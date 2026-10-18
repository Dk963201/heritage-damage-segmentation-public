import unittest
import numpy as np
from run_local import starts, clahe_rgb

class PreprocessingTests(unittest.TestCase):
    def test_edges_covered(self):
        for length in (256,257,512,5752):
            coordinates=starts(length)
            self.assertEqual(coordinates[0],0)
            self.assertEqual(coordinates[-1]+256,length)
            self.assertTrue(all(b-a<=256 for a,b in zip(coordinates,coordinates[1:])))
    def test_small_rejected(self):
        with self.assertRaises(ValueError): starts(255)
    def test_clahe_preserves_shape_and_type(self):
        image=np.random.default_rng(42).integers(0,256,(260,270,3),dtype=np.uint8)
        processed=clahe_rgb(image)
        self.assertEqual(processed.shape,image.shape)
        self.assertEqual(processed.dtype,image.dtype)
        self.assertFalse(np.array_equal(image,processed))
if __name__=='__main__':unittest.main()
