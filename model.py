import torch
import torch.nn as nn

from gcmc import GCMCLayer
from decoder import BilinearDecoder


class GCMC(nn.Module):
    def __init__(self,
                n_layers,
                edge_types,
                user_feats_dim,
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
        user_feats_dim : int
            dimension of user features
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
        self.encdoers = nn.ModuleList()
        for _ in range(n_layers):
            self.encdoers.append(GCMCLayer(edge_types = edge_types,
                                            user_feats_dim = user_feats_dim,
                                            item_feats_dim = item_feats_dim,
                                            out_feats_dim = hidden_feats_dim,
                                            agg = agg,
                                            drop_out = drop_out,
                                            activation = activation))
            user_feats_dim, item_feats_dim = hidden_feats_dim, hidden_feats_dim

            
        self.decoder = BilinearDecoder(feats_dim = out_feats_dim,
                                        n_classes = len(edge_types),
                                        n_basis = n_basis)

    def forward(self,
                enc_graph,
                dec_graph,
                ufeats,
                ifeats,
                ukey = 'user',
                ikey = 'item'):
        for encoder in self.encdoers:
            ufeats, ifeats = encoder(enc_graph, ufeats, ifeats, ukey, ikey)

        pred_edge_types = self.decoder(dec_graph, ufeats, ifeats, ukey, ikey)

        return pred_edge_types
