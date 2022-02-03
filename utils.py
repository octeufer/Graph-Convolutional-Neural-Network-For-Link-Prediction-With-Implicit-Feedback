import torch
import numpy as np
from torch.nn import functional as F

def identity_mapping(x):
    return x

activation_map = {
    'relu' : F.relu,
    'leaky' : F.leaky_relu,
    'selu' : F.selu,
    'sigmoid' : F.sigmoid,
    'tanh' : F.tanh,
    'none' : identity_mapping
}

def add_degree(graph, edge_types, symmetric = True, n_customers = None, n_items = None):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))

        return x.unsqueeze(1)

    customer_ci = []
    customer_cj = []
    product_ci = []
    product_cj = []
    for r in edge_types:
        customer_ci.append(graph[f'reverse-{r}'].in_degrees())
        product_ci.append(graph[f'{r}'].in_degrees())
        
        if symmetric:
            customer_cj.append(graph[f'{r}'].out_degrees())
            product_cj.append(graph[f'reverse-{r}'].out_degrees())

    customer_ci = _calc_norm(sum(customer_ci))
    product_ci = _calc_norm(sum(product_ci))

    if symmetric:
        customer_cj = _calc_norm(sum(customer_cj))
        product_cj = _calc_norm(sum(product_cj))
    else:
        customer_cj = torch.ones((n_customers,))
        product_cj = torch.ones((n_items,))

    graph.nodes['customer'].data.update({'ci': customer_ci, 'cj': customer_cj})
    graph.nodes['item'].data.update({'ci': product_ci, 'cj': product_cj})