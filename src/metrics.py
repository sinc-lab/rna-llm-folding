import torch
from sklearn.metrics import f1_score
from utils import mat2bp

def contact_f1(ref_batch, pred_batch, Ls, th=0.5, reduce=True, method="triangular"):
    """Compute F1 from base pairs. Input goes to sigmoid and then thresholded"""
    f1_list = []

    if type(ref_batch) == float or len(ref_batch.shape) < 3:
        ref_batch = [ref_batch]
        pred_batch = [pred_batch]
        L = [L]

    for ref, pred, l in zip(ref_batch, pred_batch, Ls):
        # ignore padding
        ind = torch.where(ref != -1)
        pred = pred[ind].view(l, l)
        ref = ref[ind].view(l, l)

        # pred goes from -inf to inf
        pred = torch.sigmoid(pred)
        pred[pred<=th] = 0

        if method == "triangular":
            f1 = f1_triangular(ref, pred>0)
        if method == "f1_shift":
            ref_bp = mat2bp(ref)
            pred_bp = mat2bp(pred)
            f1 = f1_shift(ref_bp, pred_bp)
        
        f1_list.append(f1)

    if reduce:
        return torch.tensor(f1_list).mean().item()
    else:
        return torch.tensor(f1_list)


def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(ref, pred, zero_division=0)


def f1_strict(ref_bp, pre_bp):
    """F1 score strict, same as triangular but less efficient"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if rbp in pre_bp:
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if pbp in ref_bp:
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1


def f1_shift(ref_bp, pre_bp):
    """F1 score with tolerance of 1 position"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if (
            rbp in pre_bp
            or [rbp[0], rbp[1] - 1] in pre_bp
            or [rbp[0], rbp[1] + 1] in pre_bp
            or [rbp[0] + 1, rbp[1]] in pre_bp
            or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
            pbp in ref_bp
            or [pbp[0], pbp[1] - 1] in ref_bp
            or [pbp[0], pbp[1] + 1] in ref_bp
            or [pbp[0] + 1, pbp[1]] in ref_bp
            or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1
