import torch
import torch.nn as nn

from gcmc import GCMCLayer
from decoder import BilinearDecoder

from utils import activation_map


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
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(GCMCLayer(edge_types = edge_types,
                                            user_feats_dim = user_feats_dim,
                                            item_feats_dim = item_feats_dim,
                                            out_feats_dim = hidden_feats_dim,
                                            agg = agg,
                                            drop_out = drop_out,
                                            activation = activation))
            user_feats_dim, item_feats_dim = hidden_feats_dim, hidden_feats_dim

        self.linear_user = nn.Linear(hidden_feats_dim, out_feats_dim)
        self.linear_item = nn.Linear(hidden_feats_dim, out_feats_dim)
        self.activation_out = activation_map[activation]
            
        self.decoder = BilinearDecoder(feats_dim = out_feats_dim,
                                        n_classes = len(edge_types),
                                        n_basis = n_basis)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_user.weight)
        torch.nn.init.xavier_uniform_(self.linear_item.weight)

    def forward(self,
                enc_graph,
                dec_graph,
                ufeats,
                ifeats,
                ukey = 'user',
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
            user_{i} = \sigma( W_u * h_{i} )
            item_{j} = \sigma( W_v * h_{j} )

        3. Bilinear decoder
            logits_{i, j, r} = ufeats_{i} @ Q_r @ ifeats_{j}
        """

        for encoder in self.encoders:
            ufeats, ifeats = encoder(enc_graph, ufeats, ifeats, ukey, ikey)

        ufeats = self.activation_out(self.linear_user(ufeats))
        ifeats = self.activation_out(self.linear_item(ifeats))

        pred_edge_types = self.decoder(dec_graph, ufeats, ifeats, ukey, ikey)

        return pred_edge_types
