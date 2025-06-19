import numpy as np

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def confusion_matrix(true_lbl, pred_lbl, num_classes=None):
    k = num_classes or true_lbl.max() + 1
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (true_lbl, pred_lbl), 1)
    return cm

def get_accuracy(TP, FP, FN, TN):
    return (TP) / (TP + FP + FN + TN)

def get_precision(TP, FP, FN, TN):
    return TP / (TP + FP)

def get_recall(TP, FP, FN, TN):
    return TP / (TP + FN)

def get_f1(TP, FP, FN, TN):
    precision = get_precision(TP, FP, FN, TN)
    recall = get_recall(TP, FP, FN, TN)
    return 2 * (precision * recall) / (precision + recall)

def get_fn_rate(TP, FP, FN, TN):
    return FN / (FN + TP)

def get_matrix_metrics(cm):
    true_positive = np.diag(cm)
    false_positive = cm.sum(axis=0) - true_positive
    false_negative = cm.sum(axis=1) - true_positive
    true_negative = cm.sum() - (true_positive + false_positive + false_negative)
    return true_positive, false_positive, false_negative, true_negative

def get_metrics(probs, targets):
    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(targets, axis=1)
    cm = confusion_matrix(true_labels, pred_labels)
    TP, FP, FN, TN = get_matrix_metrics(cm)
    accuracy = get_accuracy(TP, FP, FN, TN)
    precision = get_precision(TP, FP, FN, TN)
    recall = get_recall(TP, FP, FN, TN)
    f1 = get_f1(TP, FP, FN, TN)
    fn_rate = get_fn_rate(TP, FP, FN, TN)
    #per-class f1
    TPg, FPg, FNg = TP.sum(), FP.sum(), FN.sum()
    micro_prec = TPg / (TPg+FPg);  micro_rec = TPg / (TPg+FNg)
    micro_f1   = 2*micro_prec*micro_rec / (micro_prec+micro_rec)
    out = {
        "confusion_matrix": cm,
        "per_class": {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "fn_rate": fn_rate,
        },
        "macro_avg": {
            "accuracy":  accuracy.mean(),
            "precision": precision.mean(),
            "recall":    recall.mean(),
            "f1":        f1.mean(),
            "fn_rate": fn_rate.mean(),
        },
        "micro_avg": {
            "accuracy": TP.sum() / cm.sum(),
            "precision": TP.sum() / (TP+FP).sum(),
            "recall":    TP.sum() / (TP+FN).sum(),
            "f1": micro_f1,
        },
    }
    return out

def batcher(data, labels, batch_size):
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)
    for i in range(0, len(data), batch_size):
        idx = idxs[i:i+batch_size]
        yield data[idx], labels[idx]