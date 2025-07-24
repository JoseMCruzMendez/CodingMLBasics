import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from NeuralNetworks.VecOptim import SGD, MomentumSGD, AdaGrad, RMSProp, Adam, MomentumSGDW, AdaGradW, RMSPropW, AdamW
from NeuralNetworks.NpAutograd import logloss, softmax


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def confusion_matrix(true_lbl, pred_lbl, num_classes=None):
    k = num_classes or true_lbl.max() + 1
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (true_lbl, pred_lbl), 1)
    return cm

def get_accuracy(TP, FP, FN, TN):
    return TP / (TP + FP + FN + TN)

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

def cosine_annealing_schedule(epochs: int, batchsize: int, dataset_length: int, warmup_steps: float = 0.1):
    num_steps = np.ceil(dataset_length / batchsize) * epochs
    num_warmup = warmup_steps * num_steps
    def schedule(t):
        if t < num_warmup:
            return t / num_warmup
        else:
            return np.cos((t - num_warmup) / (num_steps - num_warmup) * np.pi / 2)
    return schedule

def get_mnist():
    mnist_train = pd.read_csv("data/mnist_train.csv", header=None).to_numpy(dtype=np.float32)
    train_labels, train_data = mnist_train[:,0], mnist_train[:,1:]
    train_data = train_data / 255.
    train_labels = one_hot(train_labels.astype(np.int32), 10)
    test_data = pd.read_csv("data/mnist_test.csv", header=None).to_numpy(dtype=np.float32)
    test_labels, test_data = test_data[:,0], test_data[:,1:]
    test_data = test_data / 255.
    test_labels = one_hot(test_labels.astype(np.int32), 10)
    return train_data, train_labels, test_data, test_labels

def run_test(model, datasets, optim, reg_type, reg_param, epochs=50, batch_size=256, **kwargs):
    train_data, train_labels, test_data, test_labels = datasets
    schedule = cosine_annealing_schedule(epochs, batch_size, len(train_data))
    optimizer = optim(model.get_params(), lr_func=schedule, reg_type=reg_type, reg_param=reg_param, **kwargs)
    losses = []
    for i in range(epochs):
        batch = batcher(train_data, train_labels, batch_size)
        for batch_data, batch_lbl in batch:
            logits = model(batch_data)
            loss = logloss(logits, batch_lbl)
            loss.backward()
            optimizer.step()
            losses.append(loss.v)
        if i % 10 == 0:
            print(f"Epoch {i}: Loss {loss.v}")
    stats = get_metrics(softmax(model(test_data)).v, test_labels)
    print(f"Confusion Matrix:\n{stats['confusion_matrix']}\n {stats['micro_avg']}")
    return losses, [param.v.flatten() for param in neural_net.get_params()]

def train_on_optims(model, datasets, epochs=50, w=False):
    colors = ["blue", "red", "orange", "green", "purple"]
    labels = ["SGD", "Momentum SGD", "AdaGrad", "RMSProp", "Adam"]
    optimizers = [SGD, MomentumSGD, AdaGrad, RMSProp, Adam] if not w else [SGD, MomentumSGDW, AdaGradW, RMSPropW, AdamW]
    arglist = [{"lr":1e-3},
               {"momentum":0.9, "lr":1e-1},
               {"eps":1e-12, "momentum":0.9, "lr":1e-3},
               {"eps":1e-12, "momentum":0.9, "lr":1e-3},
               {"eps":1e-12, "momentum":0.9, "beta2":0.99, "lr":1e-3}]
    reg_types = ["L1", "L2", "FLinf", "Linf"]
    reg_params = [1e-6, 1e-6, 1e-2, 1e-3]
    for reg_type, reg_param in zip(reg_types, reg_params):
        fig, axs = plt.subplots(2, figsize=(10,10), constrained_layout=True)
    for optimizer, color, label, kwargs in zip(optimizers, colors, labels, arglist):
        print(f"Running {label} with reg type {reg_type} and reg param {reg_param}")
        losses, params = run_test(model, datasets, optimizer, reg_type, reg_param, epochs=epochs, **kwargs)
        losses = np.array(losses)
        smoothed_losses = (losses.cumsum()[100:] - losses.cumsum()[:-100])/100
        axs[0].plot(smoothed_losses, color=color, label=label)
        axs[1].hist(params, color=(color, color), label=label, bins=15, density=True)
    axs[0].legend()
    axs[0].set_title(f"Loss vs Update Number")
    axs[1].legend()
    axs[1].set_title(f"Weights")
    fig.suptitle(f"Optimizer: {reg_type} Reg Param: {reg_param}")
    plt.show()

def sma(vals: np.ndarray, steps=100)->np.ndarray:
    """Short simple moving average implementation."""
    cum_sum = vals.cumsum()
    moving_average = (cum_sum[steps:]-cum_sum[:-steps])/steps
    return moving_average
