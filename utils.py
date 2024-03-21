import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import random as rand
import time
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def rbf_fn(x, c, s, w):
    return w * torch.exp((-1 * (x - c) * (x - c) / (2 * (s * s))))

def plot_rbf(model, data, target,model_list = None):
    plt.figure(figsize=(15, 6))
    t = torch.tensor(np.arange(0, len(data) - 0.99, 0.1), dtype=torch.float, device=device)
    rbf_sum_point = sum([rbf_fn(t, c, s, w) for c, s, w in zip(model.clt_list, model.std_list, model.weight_list)])
    if model_list != None:
        for i in range(len(model_list)):
            rbf_sum_point += sum(
                [rbf_fn(t, c, s, w) for c, s, w in zip(model_list[i].add_clt, model_list[i].add_std, model_list[i].add_weight)])
    plt.plot(t.cpu().detach().numpy(), rbf_sum_point.cpu().detach().numpy())  # rbf_plot

    # target plot
    plt.scatter(data.cpu().detach().numpy(), target.cpu().detach().numpy())
    plt.plot(target.cpu().detach().numpy())
    plt.show()

def save_rbf(init_model, add_model):
    clt = []
    std = []
    weight = []

    clt.append(init_model.clt_list.detach())
    std.append(init_model.std_list.detach())
    weight.append(init_model.weight_list.detach())
    
    for i in range(len(add_model)):
        clt.append(add_model[i].add_clt.detach())
        std.append(add_model[i].add_std.detach())
        weight.append(add_model[i].add_weight.detach())
        
    return clt, std, weight