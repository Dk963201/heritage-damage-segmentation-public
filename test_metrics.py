import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import segmentation_metrics


class MetricsTests(unittest.TestCase):
    def test_known_confusion(self):
        r = segmentation_metrics([[8, 2], [1, 9]])
        self.assertAlmostEqual(r['pixel_accuracy'], .85)
        self.assertAlmostEqual(r['class_iou'][0], 8 / 11)
        self.assertAlmostEqual(r['class_iou'][1], 9 / 12)
        self.assertAlmostEqual(r['class_dice'][1], 18 / 21)

    def test_absent_class_not_perfect(self):
        r = segmentation_metrics([[3, 0], [0, 0]])
        self.assertIsNone(r['class_iou'][1])
        self.assertEqual(r['mean_iou'], 1)

    def test_empty_observations(self):
        self.assertIsNone(segmentation_metrics([[0, 0], [0, 0]])['pixel_accuracy'])

    def test_bad_input(self):
        for matrix in [[], [[1, 2]], [[-1]], [[float('nan')]], [[True]]]:
            with self.assertRaises(ValueError):
                segmentation_metrics(matrix)


if __name__ == '__main__':
    unittest.main()
