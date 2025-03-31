import torch
import matplotlib.pyplot as plt


def plot_bridge_load(pred, true, contr=None):
    if len(pred.shape) == 1:
        pred = pred[:, None]
        true = true[:, None]
    f_dim = pred.shape[1]
    labels = ['Load', 'Speed']
    factors = [40000, 50]
    if len(contr.shape) == 1:
        contr = contr[:, None]
    if contr is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 2))
        for i in range(f_dim):
            true_i = true[::10, i] * factors[i]
            pred_i = pred[::10, i] * factors[i]
            ax.plot(true_i, '--', label=f'{labels[i]} true', color='k')
            ax.plot(pred_i, label=f'{labels[i]} pred', alpha=0.7)
        ax.legend()
    else:
        # check shape of contr
        c_dim = contr.shape[1]
        f, ax = plt.subplots(f_dim+c_dim, 1, figsize=(8, 2*(f_dim+c_dim)))
        if f_dim == 1:
            labels = ['Load']
        for i in range(f_dim):
            true_i = true[::10, i] * factors[i]
            pred_i = pred[::10, i] * factors[i]
            ax[i].plot(true_i, '--', label=f'{labels[i]} true', color='k')
            ax[i].plot(pred_i, label=f'{labels[i]} pred', alpha=0.7)
            ax[i].legend()
        contr_labels = ['temperature', 'speed']
        for i in range(c_dim):
            ax[i+f_dim].plot(contr[::10, i], label=contr_labels[i])
            ax[i+f_dim].legend()
    return f


def plot_bearing_load(pred, true, rot=None):
    # if dim of pred is 1 extend it to 2
    if len(pred.shape) == 1:
        pred = pred[:, None]
        true = true[:, None]
    # check if pred is a list of tensors
    if isinstance(pred, list):
        true, pred = torch.cat(true).cpu().numpy(), torch.cat(pred).cpu().numpy()
    f_dim = pred.shape[1]
    labels = ['Fx', 'Fz']
    factors = [8000, 1000]
    p_dim = f_dim+1 if rot is not None else f_dim
    f, axes = plt.subplots(p_dim, 1, figsize=(10, 3*p_dim))
    if f_dim == 1:
        labels = ['Ft']
        axes = [axes]
    for i in range(f_dim):
        true_i = true[:, i] * factors[i]
        pred_i = pred[:, i] * factors[i]
        axes[i].plot(true_i, '--', label=f'{labels[i]} true', color='k')
        axes[i].plot(pred_i, label=f'{labels[i]} pred', alpha=0.7)
        axes[i].legend()
    if rot is not None:
        axes[-1].plot(rot, label='rot')
    return f

