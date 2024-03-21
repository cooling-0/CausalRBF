import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import random as rand
import time
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class RBFnet(nn.Module):
    def __init__(self, rbf_num):
        super(RBFnet, self).__init__()

        self.rbf_num = rbf_num
        self.std_list = nn.Parameter(torch.rand(self.rbf_num))
        self.weight_list = nn.Parameter(torch.rand(self.rbf_num))
        self.clt_list = nn.Parameter(torch.rand(self.rbf_num))

    def rbf(self, X, clt, std, w):
        return w * torch.exp((-1 * (X - clt) * (X - clt) / (2 * (std * std))))

    def forward(self, X):
        y = sum([self.rbf(X, c, s, w) for c, s, w in zip(self.clt_list, self.std_list, self.weight_list)])

        return y


class Add_rbf(nn.Module):
    def __init__(self, clt):
        super(Add_rbf, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clt = clt
        self.add_clt = nn.Parameter(torch.tensor(self.clt, dtype= torch.float, device= device))
        self.add_std = nn.Parameter(torch.rand(len(self.clt), device=device))
        self.add_weight = nn.Parameter(torch.rand(len(self.clt), device=device))

    def rbf(self, X, clt, std, w):
        return w * torch.exp((-1 * (X - clt) * (X - clt) / (2 * (std * std))))

    def forward(self, x):
        x = sum([self.rbf(x, c, s, w) for c, s, w in zip(self.add_clt, self.add_std, self.add_weight)])

        return x

def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def Init_train(model, X, Y, lr, epochs, device):
    model.to(device)

    loss_list = []
    F_list = []

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_it = None
    best_model = None
    best_loss = np.inf

    for epoch in range(epochs):
        y = model(X)
        F_list.append(y)

        loss = sum([loss_fn(y[i], Y[i]) for i in range(len(y))])
        loss_list.append(loss)

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print('{} epoch train'.format(epoch))
            print('loss :', loss)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        if loss < best_loss:
            best_loss = loss
            best_it = epoch
            best_model = deepcopy(model)

    print('best epoch : {}, best_loss : {}'.format(best_it, best_loss))
    restore_parameters(model, best_model)

    print('-------------------------------------------------------------------------')
    print()

    clt_list = best_model.clt_list.clone().detach()
    std_list = best_model.std_list.clone().detach()
    weight_list = best_model.weight_list.clone().detach()

    return best_model, clt_list, std_list, weight_list


def mulit_rbf_clt(mulit_rbf_num, target):
    clt = []
    for i in range(mulit_rbf_num):
        index_ = torch.argmax(abs(target)).cpu().detach().tolist()
        target[index_] = 0
        clt.append(index_)

    return clt

def add_train(init_best_model, X, Y, mulit_rbf_num, epochs, lr, loss_th, device):
    init_best_model.to(device)
    init_y = Y - init_best_model(X)

    add_list = nn.ModuleList()
    target_list = []

    best_loss = np.inf
    add = 0
    while best_loss > loss_th:
        if add == 0:
            target = init_y.clone().detach()
            clt = mulit_rbf_clt(mulit_rbf_num, init_y)
        else:
            target = add_y.clone().detach()
            clt = mulit_rbf_clt(mulit_rbf_num, add_y)

        target_list.append(target)
        print('{}th add rbf'.format(add))

        loss_fn = nn.MSELoss()

        best_it = None
        best_model = None
        best_loss = np.inf

        model = Add_rbf(clt)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            y = model(X)

            loss = sum([loss_fn(y[i], target[i]) for i in range(len(y))])

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print('{} epoch train'.format(epoch))
                print('loss :', loss)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if loss < best_loss:
                best_loss = loss
                best_it = epoch
                best_model = deepcopy(model)

        add_list.append(best_model)
        add_y = target - best_model(X)
        add += 1
        print('best epoch : {}, best_loss : {}'.format(best_it, best_loss))

    return add_list, target_list


def rbf_fn(x, c, s, w):
    return w * torch.exp((-1 * (x - c) * (x - c) / (2 * (s * s))))


def plot_rbf(model, data, target, model_list = None):
    plt.figure(figsize=(15, 6))
    t = torch.tensor(np.arange(0, len(data) - 0.99, 0.1), dtype=torch.float, device=device)
    rbf_sum_point = sum([rbf_fn(t, c, s, w) for c, s, w in zip(model.clt_list, model.std_list, model.weight_list)])
    if model_list != None:
        for i in range(len(model_list)):
            rbf_sum_point += sum(
                [rbf_fn(p, c, s, w) for c, s, w in zip(model_list[i].add_clt, model_list[i].add_std, model_list[i].add_weight)])
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
