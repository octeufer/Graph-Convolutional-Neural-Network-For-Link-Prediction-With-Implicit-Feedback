import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

def to_etype_name(purchase):
    return str(purchase).replace('.', '_')

class ASOS(object):
    """ASOS dataset used by GCMC model
    TODO(minjie): make this dataset more general
    The dataset stores ASOS purchases in two types of graphs. The encoder graph
    contains purchase information in the form of edge types. The decoder graph
    stores plain customer-product pairs in the form of a bipartite graph with no purchase info
    information. All graphs have two types of nodes: "customer" and "product".
    The training, validation and test set can be summarized as follows:
    training_enc_graph : training customer-product pairs + purchase info
    training_dec_graph : training customer-product pairs
    valid_enc_graph : training customer-product pairs + purchase info
    valid_dec_graph : validation customer-product pairs
    test_enc_graph : training customer-product pairs + validation customer-product pairs + purchase info
    test_dec_graph : test customer-product pairs
    Attributes
    ----------
    train_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for training.
    train_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for training.
    train_labels : torch.Tensor
        The categorical label of each customer-product pair
    train_truths : torch.Tensor
        The actual purchase values of each customer-product pair
    valid_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for validation.
    valid_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for validation.
    valid_labels : torch.Tensor
        The categorical label of each customer-product pair
    valid_truths : torch.Tensor
        The actual purchase values of each customer-product pair
    test_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for test.
    test_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for test.
    test_labels : torch.Tensor
        The categorical label of each customer-product pair
    test_truths : torch.Tensor
        The actual purchase values of each customer-product pair
    customer_feature : torch.Tensor
        Customer feature tensor. If None, representing an identity matrix.
    product_feature : torch.Tensor
        Product feature tensor. If None, representing an identity matrix.
    possible_purchase_values : np.ndarray
        Available purchase values in the dataset
    Parameters
    ----------
    name : str
        Dataset name.
    device : torch.device
        Device context
    mix_cpu_gpu : boo, optional
        If true, the ``customer_feature`` attribute is stored in CPU
    use_one_hot_fea : bool, optional
        If true, the ``customer_feature`` attribute is None, representing an one-hot identity
        matrix. (Default: False)
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    test_ratio : float, optional
        Ratio of test data
    valid_ratio : float, optional
        Ratio of validation data
    """
    def __init__(self, name, device, mix_cpu_gpu=False,
                 use_one_hot_fea=False, symm=True,
                 test_ratio=0.1, valid_ratio=0.1):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        self._dir = os.path.join('C:\Workspace\ASOS_TechTest')
        print("Starting processing {} ...".format(self._name))
        col_names = ['customerId', 'productId', 'purchased', 'isFemale', 'country', 'yearOfBirth', \
            'isPremier', 'brand', 'price', 'productType', 'onSale', 'dateOnSite']
        self.all_train_info = pd.read_csv(os.path.join(self._dir, 'trainset_sample.csv'), sep=',', header=None,
                                    names=col_names, engine='python')

        col_names_test = ['customerId', 'productId', 'isFemale', 'country', 'yearOfBirth', \
            'isPremier', 'brand', 'price', 'productType', 'onSale', 'dateOnSite']
        self.all_pred_test_info = pd.read_csv(os.path.join(self._dir, 'testset_sample.csv'), sep=',', header=None,
                                    names=col_names_test, engine='python')

        self.customer_info = self.all_train_info[['customerId', 'isFemale', 'country', 'yearOfBirth', 'isPremier']].drop_duplicates()
        self.product_info = self.all_train_info[['productId', 'brand', 'price', 'productType', 'onSale', 'dateOnSite']].drop_duplicates()
        print('......')

        self.all_purchase_info = self.all_train_info[['customerId', 'productId', 'purchased']]
        num_test = int(np.ceil(self.all_purchase_info.shape[0] * self._test_ratio))
        shuffled_idx = np.random.permutation(self.all_purchase_info.shape[0])
        self.test_purchase_info = self.all_purchase_info.iloc[shuffled_idx[:num_test]]
        self.all_train_purchase_info = self.all_purchase_info.iloc[shuffled_idx[num_test: ]]

        self.pred_purchase = self.all_pred_test_info[['customerId', 'productId']]

        print('......')
        num_valid = int(np.ceil(self.all_train_purchase_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_purchase_info.shape[0])
        self.valid_purchase_info = self.all_train_purchase_info.iloc[shuffled_idx[: num_valid]]
        self.train_purchase_info = self.all_train_purchase_info.iloc[shuffled_idx[num_valid: ]]
        self.possible_purchase_values = np.unique(self.train_purchase_info["purchased"].values)

        print("All purchase pairs : {}".format(self.all_purchase_info.shape[0]))
        print("\tAll train purchase pairs : {}".format(self.all_train_purchase_info.shape[0]))
        print("\t\tTrain purchase pairs : {}".format(self.train_purchase_info.shape[0]))
        print("\t\tValid purchase pairs : {}".format(self.valid_purchase_info.shape[0]))
        print("\tTest purchase pairs  : {}".format(self.test_purchase_info.shape[0]))
        print("\tpred Test purchase pairs  : {}".format(self.pred_purchase.shape[0]))

        # Map customer/product to the global id
        self.global_customer_id_map = {ele: i for i, ele in enumerate(self.customer_info['customerId'])}
        self.global_product_id_map = {ele: i for i, ele in enumerate(self.product_info['productId'])}
        print('Total customer number = {}, product number = {}'.format(len(self.global_customer_id_map),
                                                                 len(self.global_product_id_map)))
        self._num_customer = len(self.global_customer_id_map)
        self._num_product = len(self.global_product_id_map)

        ### Generate features
        if use_one_hot_fea:
            self.customer_feature = None
            self.product_feature = None
        else:
            # if mix_cpu_gpu, we put features in CPU
            if mix_cpu_gpu:
                self.customer_feature = th.FloatTensor(self._process_customer_fea())
                self.product_feature = th.FloatTensor(self._process_product_fea())
            else:
                self.customer_feature = th.FloatTensor(self._process_customer_fea()).to(self._device)
                self.product_feature = th.FloatTensor(self._process_product_fea()).to(self._device)

        if self.customer_feature is None:
            self.customer_feature_shape = (self.num_customer, self.num_customer)
            self.product_feature_shape = (self.num_product, self.num_product)
        else:
            self.customer_feature_shape = self.customer_feature.shape
            self.product_feature_shape = self.product_feature.shape
        info_line = "Feature dim: "
        info_line += "\ncustomer: {}".format(self.customer_feature_shape)
        info_line += "\nproduct: {}".format(self.product_feature_shape)
        print(info_line)

        all_train_purchase_pairs, all_train_purchase_values = self._generate_pair_value(self.all_train_purchase_info)
        train_purchase_pairs, train_purchase_values = self._generate_pair_value(self.train_purchase_info)
        valid_purchase_pairs, valid_purchase_values = self._generate_pair_value(self.valid_purchase_info)
        test_purchase_pairs, test_purchase_values = self._generate_pair_value(self.test_purchase_info)
        pred_test_pairs = self._generate_pair_value_pred(self.pred_purchase)

        def _make_labels(purchases):
            labels = th.LongTensor(np.searchsorted(self.possible_purchase_values, purchases)).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(train_purchase_pairs, train_purchase_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_purchase_pairs)
        self.train_labels = _make_labels(train_purchase_values)
        self.train_truths = th.FloatTensor(train_purchase_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_purchase_pairs)
        self.valid_labels = _make_labels(valid_purchase_values)
        self.valid_truths = th.FloatTensor(valid_purchase_values).to(device)

        self.test_enc_graph = self._generate_enc_graph(all_train_purchase_pairs, all_train_purchase_values, add_support=True)
        self.test_dec_graph = self._generate_dec_graph(test_purchase_pairs)
        self.test_labels = _make_labels(test_purchase_values)
        self.test_truths = th.FloatTensor(test_purchase_values).to(device)

        self.pred_test_enc_graph = self._generate_enc_graph(all_train_purchase_pairs, all_train_purchase_values, add_support=True)
        self.pred_test_dec_graph = self._generate_dec_graph(pred_test_pairs)

        def _npairs(graph):
            rst = 0
            for r in self.possible_purchase_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('customer'), self.train_enc_graph.number_of_nodes('item'),
            _npairs(self.train_enc_graph)))
        print("Train dec graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('customer'), self.train_dec_graph.number_of_nodes('item'),
            self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('customer'), self.valid_enc_graph.number_of_nodes('item'),
            _npairs(self.valid_enc_graph)))
        print("Valid dec graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('customer'), self.valid_dec_graph.number_of_nodes('item'),
            self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('customer'), self.test_enc_graph.number_of_nodes('item'),
            _npairs(self.test_enc_graph)))
        print("Test dec graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('customer'), self.test_dec_graph.number_of_nodes('item'),
            self.test_dec_graph.number_of_edges()))
        print("Pred enc graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.pred_test_enc_graph.number_of_nodes('customer'), self.pred_test_enc_graph.number_of_nodes('item'),
            _npairs(self.pred_test_enc_graph)))
        print("Pred dec graph: \t#customer:{}\t#product:{}\t#pairs:{}".format(
            self.pred_test_dec_graph.number_of_nodes('customer'), self.pred_test_dec_graph.number_of_nodes('item'),
            self.pred_test_dec_graph.number_of_edges()))

    def _update_global_idmap(self):
        for ele in self.pred_purchase['customerId'].unique():
            if ele not in self.global_customer_id_map:
                self.global_customer_id_map.update({ele: self._num_customer})
                self._num_customer += 1
        for ele in self.pred_purchase['productId'].unique():
            if ele not in self.global_product_id_map:
                self.global_product_id_map.update({ele: self._num_product})
                self._num_product += 1
        
        add_customer_info = self.all_pred_test_info[['customerId', 'isFemale', 'country', 'yearOfBirth', 'isPremier']].drop_duplicates()
        add_product_info = self.all_pred_test_info[['productId', 'brand', 'price', 'productType', 'onSale', 'dateOnSite']].drop_duplicates()

        self.customer_info = self.customer_info.append(add_customer_info, ignore_index=True).drop_duplicates()
        self.product_info = self.product_info.append(add_product_info, ignore_index=True).drop_duplicates()

        ### Generate features
        self.customer_feature = th.FloatTensor(self._process_customer_fea()).to(self._device)
        self.product_feature = th.FloatTensor(self._process_product_fea()).to(self._device)

        if self.customer_feature is None:
            self.customer_feature_shape = (self.num_customer, self.num_customer)
            self.product_feature_shape = (self.num_product, self.num_product)
        else:
            self.customer_feature_shape = self.customer_feature.shape
            self.product_feature_shape = self.product_feature.shape
        info_line = "Add Feature dim: "
        info_line += "\ncustomer: {}".format(self.customer_feature_shape)
        info_line += "\nproduct: {}".format(self.product_feature_shape)
        print(info_line)

        return 

    def _generate_pair_value(self, purchase_info):
        purchase_pairs = (np.array([self.global_customer_id_map[ele] for ele in purchase_info["customerId"]],
                                 dtype=np.int64),
                        np.array([self.global_product_id_map[ele] for ele in purchase_info["productId"]],
                                 dtype=np.int64))
        purchase_values = purchase_info["purchased"].values.astype(np.float32)
        return purchase_pairs, purchase_values
    
    def _generate_pair_value_pred(self, purchase_info):
        self._update_global_idmap()
        purchase_pairs = (np.array([self.global_customer_id_map[ele] for ele in purchase_info["customerId"]],
                                 dtype=np.int64),
                        np.array([self.global_product_id_map[ele] for ele in purchase_info["productId"]],
                                 dtype=np.int64))
        # purchase_values = purchase_info["purchased"].values.astype(np.float32)
        return purchase_pairs

    def _generate_enc_graph(self, purchase_pairs, purchase_values, add_support=False):
        customer_product_R = np.zeros((self._num_customer, self._num_product), dtype=np.float32)
        customer_product_R[purchase_pairs] = purchase_values

        data_dict = dict()
        num_nodes_dict = {'customer': self._num_customer, 'item': self._num_product}
        purchase_row, purchase_col = purchase_pairs
        for purchase in self.possible_purchase_values:
            ridx = np.where(purchase_values == purchase)
            rrow = purchase_row[ridx]
            rcol = purchase_col[ridx]
            purchase = to_etype_name(purchase)
            data_dict.update({
                ('customer', str(purchase), 'item'): (rrow, rcol),
                ('item', 'reverse-%s' % str(purchase), 'customer'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(purchase_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)
            customer_ci = []
            customer_cj = []
            product_ci = []
            product_cj = []
            for r in self.possible_purchase_values:
                r = to_etype_name(r)
                customer_ci.append(graph['reverse-%s' % r].in_degrees())
                product_ci.append(graph[r].in_degrees())
                if self._symm:
                    customer_cj.append(graph[r].out_degrees())
                    product_cj.append(graph['reverse-%s' % r].out_degrees())
                else:
                    customer_cj.append(th.zeros((self.num_customer,)))
                    product_cj.append(th.zeros((self.num_product,)))
            customer_ci = _calc_norm(sum(customer_ci))
            product_ci = _calc_norm(sum(product_ci))
            if self._symm:
                customer_cj = _calc_norm(sum(customer_cj))
                product_cj = _calc_norm(sum(product_cj))
            else:
                customer_cj = th.ones(self.num_customer,)
                product_cj = th.ones(self.num_product,)
            graph.nodes['customer'].data.update({'ci' : customer_ci, 'cj' : customer_cj})
            graph.nodes['item'].data.update({'ci' : product_ci, 'cj' : product_cj})

        return graph

    def _generate_dec_graph(self, purchase_pairs):
        ones = np.ones_like(purchase_pairs[0])
        customer_product_purchases_coo = sp.coo_matrix(
            (ones, purchase_pairs),
            shape=(self.num_customer, self.num_product), dtype=np.float32)
        g = dgl.bipartite_from_scipy(customer_product_purchases_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('customer', 'purchase', 'item'): g.edges()}, 
                               num_nodes_dict={'customer': self.num_customer, 'item': self.num_product})

    @property
    def num_links(self):
        return self.possible_purchase_values.size

    @property
    def num_customer(self):
        return self._num_customer

    @property
    def num_product(self):
        return self._num_product

    def _process_customer_fea(self):
        """
        Parameters
        ----------
        customer_info : pd.DataFrame
        name : str
        Returns
        -------
        customer_features : np.ndarray
        """
        nn = self.customer_info.shape[0]
        gender = self.customer_info['isFemale'].values.astype(np.float32)
        country = self.customer_info['country'].values.astype(np.float32)
        year = self.customer_info['yearOfBirth'].values.astype(np.float32)
        premier = self.customer_info['isPremier'].values.astype(np.float32)
        customer_features = np.concatenate([gender.reshape((nn, 1)), country.reshape((nn, 1)),
                                        year.reshape((nn, 1)), premier.reshape((nn, 1))], axis=1)
        return customer_features

    def _process_product_fea(self):
        """
        Parameters
        ----------
        product_info : pd.DataFrame
        name :  str
        Returns
        -------
        product_features : np.ndarray
            Generate product features by concatenating embedding and the year
        """
        nn = self.product_info.shape[0]
        brand = self.product_info['brand'].values.astype(np.float32)
        price = self.product_info['price'].values.astype(np.float32)
        type = self.product_info['productType'].values.astype(np.float32)
        sale = self.product_info['onSale'].values.astype(np.float32)
        dates = self.product_info['dateOnSite'].values.astype(np.float32)
        product_features = np.concatenate([brand.reshape((nn, 1)), price.reshape((nn, 1)),
                                        type.reshape((nn, 1)), sale.reshape((nn, 1)),
                                        dates.reshape((nn,1))], axis=1)

        return product_features

if __name__ == '__main__':
    ASOS("ASOS", device=th.device('cpu'), symm=True)
