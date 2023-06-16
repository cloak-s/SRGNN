import numpy as np
import scipy.sparse as sp
import torch
import random
import data.io as io
import copy
import torch.nn.functional as F
import seaborn as sns
import numba
import math
# import resource
from scipy.linalg import expm
from sklearn.metrics import f1_score

def get_Node2_top_k_node(adj, idx, k, degree):
    all_neighbors = []
    for node in idx:
        r1_node = sp.find(adj[node])[1]
        r1_degrees = []
        for nodes in r1_node:
            r1_degrees.append(degree[nodes])
        sort_degree = np.argsort(r1_degrees)[::-1]
        r1_node = r1_node[sort_degree]
        if len(r1_node) < k:
            r2_node = set()
            for i in r1_node:
                r2_node.update(sp.find(adj[i])[1])
            if r2_node:
                r2_node = r2_node - set(list(r1_node))
                if node in r2_node:  # 去除中心节点
                    r2_node.remove(node)
            if r2_node:
                r2_node = np.array((list(r2_node)))
                r2_degrees = []
                for nodes in r2_node:
                    r2_degrees.append(degree[nodes])
                sort_degrees = np.argsort(r2_degrees)[::-1]
                r2_node = r2_node[sort_degrees]

                if len(r2_node) < k - len(r1_node):
                    temp = np.full((1, k), -1)[0]
                    r2_node = np.hstack((r2_node, temp))
            else:
                r2_node = np.full((1, k), -1)[0]
            neighbor = np.hstack((r1_node, r2_node))[:k]
            neighbor = torch.tensor(neighbor)
        else:
            neighbor = r1_node[:k]
            neighbor = torch.tensor(neighbor)
        all_neighbors.append(neighbor)
    return torch.stack(all_neighbors, dim=0)


def get_map_matrix(neighbor_matrix, idx, features, gama):
    all_map = []
    node_feat = torch.zeros(features.shape[1])
    for index, central_node in enumerate(idx):
        temp_map = []
        for neighbor_node in neighbor_matrix[index][:]:
            if neighbor_node != -1:
                temp_map.append(features[neighbor_node])
            else:
                temp_map.append(node_feat)
        neighbor_map = torch.stack(temp_map, dim=0)
        map_matrix = (1 - gama) * neighbor_map + gama * features[central_node]
        all_map.append(map_matrix)
    return torch.stack(all_map, dim=0)


def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719):
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx


def exclude_idx(idx: np.ndarray, idx_exclude_list):
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114):
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
            idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
        exclude_idx(idx, [train_idx]),
        nstopping, replace=False)
    return train_idx, stopping_idx


def gen_splits(labels: np.ndarray, idx_split_args,
               test: bool = False):
    labels = labels.cpu()
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
        all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, val_idx = train_stopping_split(
        known_idx, labels[known_idx], **stopping_split_args)
    if test:
        test_idx = unknown_idx
    else:
        test_idx = exclude_idx(known_idx, [train_idx, val_idx])
    return train_idx, val_idx, test_idx


def get_data(graph_name='cora_ml'):
    dataset = io.load_dataset(graph_name)
    dataset.standardize(select_lcc=True)
    features = dataset.attr_matrix
    features = normalize_features(features)
    features = np.array(features.todense())
    # features = torch.FloatTensor(features)
    labels = dataset.labels
    adj = dataset.adj_matrix
    adj_loop = adj + sp.eye(adj.shape[0])
    sys_adj = normalize_sys_adj(adj_loop)
    # sys_adj = random_walk_adj_matrix(adj_loop)
    sys_adj = sparse_mx_to_torch_sparse_tensor(sys_adj)
    labels = torch.LongTensor(labels)
    return features, adj, labels, sys_adj


def get_reddit(normalize=True, self_loop=False, graph_name='reddit'):
    dataset = io.load_dataset(graph_name)
    dataset.standardize(select_lcc=True)
    features = dataset.attr_matrix
    adj = dataset.adj_matrix
    if normalize:
        features = normalize_features(features)
    if self_loop:
        adj_loop = adj + sp.eye(adj.shape[0])  # 加入自循环
        sys_adj = normalize_sys_adj(adj_loop)
        sys_adj = sparse_mx_to_torch_sparse_tensor(sys_adj)
    labels = torch.LongTensor(dataset.labels)
    return features, adj, labels


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_mask(graph_name, seed, labels, train_per_class, val_per_class, Fixed_total=True):
    if graph_name == 'cora_ml' or graph_name == 'citeseer' or graph_name == 'pubmed':
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': seed}
        idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)
    else:
        if Fixed_total:
            idx_train, idx_val, idx_test = Fixed_Total(seed, labels, train_per_class, val_per_class)
        else:
            idx_train, idx_val, idx_test = Per_Class(seed, labels, train_per_class, val_per_class)
    idx_train = idx_train.astype(np.int64)
    idx_val = idx_val.astype(np.int64)
    idx_test = idx_test.astype(np.int64)
    return idx_train, idx_val, idx_test


def Per_Class(seed, labels, train_per, val_per):
    labels = labels.cpu()
    rnd_state = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    train_idx = []
    val_idx = []
    for i in range(max(labels)+1):
        temp_idx = idx[labels == i]
        train_idx.append(rnd_state.choice(temp_idx, train_per, replace=False))
        res_idx = exclude_idx(temp_idx, [train_idx])
        val_idx.append(rnd_state.choice(res_idx, val_per, replace=False))
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    known_idx = np.concatenate((train_idx, val_idx))
    test_idx = exclude_idx(idx, [known_idx])
    return train_idx, val_idx, test_idx


def Fixed_Total(seed, labels, train_per, val_per):
    np.random.seed(seed)
    num_node = len(labels)
    num_class = max(labels) + 1
    rnd = np.random.permutation(num_node)
    train_idx = np.sort(rnd[: train_per * num_class])
    val_idx = np.sort(rnd[train_per * num_class: train_per * num_class + val_per * num_class])
    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(num_node), train_val_idx))
    return train_idx, val_idx, test_idx


def get_idx_train_val(seed, labels, development, per_class):
    labels = labels.cpu()
    rnd_state = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    train_idx = []
    class_num = max(labels)+1

    development_idx = rnd_state.choice(idx, development, replace=False)
    test_idx = exclude_idx(idx, [development_idx])
    for i in range(class_num):
        temp_idx = development_idx[labels[development_idx] == i]
        train_idx.append(rnd_state.choice(temp_idx, per_class, replace=False))
    train_idx = np.concatenate(train_idx)
    val_idx = exclude_idx(development_idx, [train_idx])

    train_idx = train_idx.astype(np.int64)
    val_idx = val_idx.astype(np.int64)
    test_idx = test_idx.astype(np.int64)
    return train_idx, val_idx, test_idx


def normalize_sys_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def random_walk_adj_matrix(mx):  # DA
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# def get_ppr_matrix(
#         adj_matrix: np.ndarray,
#         alpha: float = 0.1) -> np.ndarray:
#     num_nodes = adj_matrix.shape[0]
#     A_tilde = adj_matrix + np.eye(num_nodes)
#     D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
#     H = D_tilde @ A_tilde @ D_tilde   # 对称归一化转移矩阵T
#     return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)



def get_PPR_matrix(norm_adj, alpha):
    A_inner = torch.eye(norm_adj.shape[0]) - (1 - alpha) * norm_adj
    ppr_matrix = alpha * torch.inverse(A_inner)
    return ppr_matrix


def cal_heat_kernel_matrix(adj, t):
    adj_matrix = np.array(adj.todense())
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))


def print_num_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("the total number of params----->", num_params)


def bootstrapping(acc):
    boot_series = sns.algorithms.bootstrap(acc, func=np.mean, n_boot=1000)
    acc_ci = np.max(np.abs(sns.utils.ci(boot_series, 95) - np.mean(acc)))
    mean_acc = np.mean(acc)
    uncertainty = np.mean(acc_ci)
    print(f"bootstrap -->Mean accuracy: {100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%")


def mix_order_features(adj, feature, k, alpha):
    feature_ori = alpha * feature
    order_feature = feature
    mix_sum = 0
    for _ in range(k):
        order_feature = torch.matmul(adj, order_feature)
        mix_sum += 1/k * ((1-alpha)*order_feature + feature_ori)
    return mix_sum


def each_order_features(adj, feature, k):
    features = []
    features.append(feature)
    order_feature = feature
    for _ in range(k):
        order_feature = torch.spmm(adj, order_feature)
        features.append(order_feature)
    return features


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#
# def get_max_memory_bytes():  # just for linux
#     return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

