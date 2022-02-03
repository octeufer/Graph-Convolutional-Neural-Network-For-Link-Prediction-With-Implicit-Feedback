import torch
import torch.nn as nn

import torch.nn.functional as F

from gcmc import GCMCLayer
from decoder import BilinearDecoder

from utils import activation_map


class GCMC(nn.Module):
    def __init__(self,
                n_layers,
                edge_types,
                customer_feats_dim,
                item_feats_dim,
                hidden_feats_dim,
                out_feats_dim,
                agg,
                drop_out,
                activation,
                n_basis):
        super().__init__()
        """Graph Convolutional Matrix Completion
        paper : https://arxiv.org/pdf/1706.02263.pdf

        n_layers : int
            number of GCMC layers
        edge_types : list
            all edge types
        customer_feats_dim : int
            dimension of customer features
        item_feats_dim : int
            dimension of item features
        hidden_feats_dim : int
            dimension of hidden features
        out_feats_dim : int
            dimension of output features
        agg : str
            aggreration type
        activation : str
            activation function
        n_basis : int
            number of basis ( <= n_classes )
        """
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(GCMCLayer(edge_types = edge_types,
                                            customer_feats_dim = customer_feats_dim,
                                            item_feats_dim = item_feats_dim,
                                            out_feats_dim = hidden_feats_dim,
                                            agg = agg,
                                            drop_out = drop_out,
                                            activation = activation))
            customer_feats_dim, item_feats_dim = hidden_feats_dim, hidden_feats_dim

        self.linear_customer = nn.Linear(hidden_feats_dim, out_feats_dim)
        self.linear_item = nn.Linear(hidden_feats_dim, out_feats_dim)
        self.activation_out = activation_map[activation]
            
        self.decoder = BilinearDecoder(feats_dim = out_feats_dim,
                                        n_classes = len(edge_types),
                                        n_basis = n_basis)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_customer.weight)
        torch.nn.init.xavier_uniform_(self.linear_item.weight)

    def forward(self,
                enc_graph,
                dec_graph,
                ufeats,
                ifeats,
                ukey = 'customer',
                ikey = 'item'):
        """
        Parameters
        ----------
        enc_graph : dgl.graph
        dec_graph : dgl.homograph

        Notes
        -----
        1. GCMC encoder (GAE ; Graph AutoEncoder)
            1) message passing
                MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
            2) aggregation
                h_{i} = \sigma( aggregate( MP_{i} ) )

        2. final features
            customer_{i} = \sigma( W_u * h_{i} )
            item_{j} = \sigma( W_v * h_{j} )

        3. Bilinear decoder
            logits_{i, j, r} = ufeats_{i} @ Q_r @ ifeats_{j}
        """

        for encoder in self.encoders:
            ufeats, ifeats = encoder(enc_graph, ufeats, ifeats, ukey, ikey)

        ufeats = self.activation_out(self.linear_customer(ufeats))
        ifeats = self.activation_out(self.linear_item(ifeats))

        pred_edge_types = self.decoder(dec_graph, ufeats, ifeats, ukey, ikey)

        return pred_edge_types


class MLP(nn.Module):
    def __init__(self,
                n_layers,
                customer_feats_dim,
                item_feats_dim,
                hidden_feats_dim,
                n_basis):
        super().__init__()

        self.n_layers = n_layers
        self.input = nn.Linear(customer_feats_dim + item_feats_dim, hidden_feats_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_feats_dim, hidden_feats_dim) for _ in range(n_layers-1)])
        self.output = nn.Linear(hidden_feats_dim, n_basis)
        self.act = nn.ELU(inplace=True)
    
    def forward(self,
                ufeats,
                ifeats):
        z = self.act(self.input(torch.cat((ufeats,ifeats),dim=1)))
        for i in range(self.n_layers-1):
            z = self.act(self.hidden[i](z))
        out = self.act(self.output(z))
        return out