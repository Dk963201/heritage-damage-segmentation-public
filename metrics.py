"""Segmentation scores from a confusion matrix (rows truth, columns prediction)."""
import math


def segmentation_metrics(matrix):
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError('Expected a nonempty square confusion matrix')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or x < 0 for row in matrix for x in row):
        raise ValueError('Counts must be finite nonnegative numbers')
    total = sum(map(sum, matrix))
    iou, dice = [], []
    for c in range(n):
        tp = matrix[c][c]
        truth = sum(matrix[c])
        predicted = sum(row[c] for row in matrix)
        union = truth + predicted - tp
        iou.append(tp / union if union else None)
        dice.append(2 * tp / (truth + predicted) if truth + predicted else None)
    supported = [x for x in iou if x is not None]
    return {'pixel_accuracy': sum(matrix[c][c] for c in range(n)) / total if total else None,
            'class_iou': iou, 'class_dice': dice,
            'mean_iou': sum(supported) / len(supported) if supported else None}
