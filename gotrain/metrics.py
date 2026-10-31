"""Dataset-wide confusion metrics; IoU is not averaged over mini-batches."""
import numpy as np


def confusion_matrix(target, prediction, classes):
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError("Prediction and target shapes differ")
    if np.any((target < 0) | (target >= classes)):
        raise ValueError("Unexpected target class; define an explicit ignore policy first")
    if np.any((prediction < 0) | (prediction >= classes)):
        raise ValueError("Unexpected prediction class")
    return np.bincount(
        classes * target.astype(np.int64) + prediction.astype(np.int64),
        minlength=classes * classes,
    ).reshape(classes, classes)


def scores(matrix):
    cm = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall[support > 0].mean()) if np.any(support > 0) else 0.0,
        "macro_f1": float(f1.mean()),
        "mean_iou": float(iou[union > 0].mean()) if np.any(union > 0) else 0.0,
        "foreground_iou": float(iou[1]) if len(iou) == 2 else None,
        "per_class_iou": iou.tolist(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }
