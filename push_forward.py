import numba
import numpy as np
import scipy.sparse as sp
from numba import prange


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = 1.0
    q = [inode]
    while len(q) > 0:
        vnode = q.pop()
        res = r[vnode] if vnode in r else f32_0
        if vnode in p:
            p[vnode] += alpha * res
        else:
            p[vnode] = alpha * res
        r[vnode] = f32_0
        _val = (1 - alpha) * res / deg[vnode]
        for unode in indices[indptr[vnode]:indptr[vnode + 1]]:
            if unode in r:
                r[unode] += _val
            else:
                r[unode] = _val
            res_unode = r[unode] if unode in r else f32_0
            if res_unode >= epsilon * deg[unode]:
                if unode not in q:
                    q.append(unode)
    return list(p.keys()), list(p.values())


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""
    out_degree = np.sum(adj_matrix > 0, axis=1).A1

    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk)
    # return neighbors

    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, normalization='row'):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    # neighbor_idx = ppr_topk(adj_matrix, alpha, eps, idx, topk)
    topk_matrix = ppr_topk(adj_matrix, alpha, eps, idx, topk)
    topk_matrix = topk_matrix.tocsr()

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")
    return topk_matrix


@numba.njit(cache=True, parallel=True, fastmath=True)
def feature_fusion(indices, indptr, central_idx, gama, features):
    all_map = np.zeros(shape=(len(central_idx), features.shape[1]))
    for i in prange(len(central_idx)):
        central_node = central_idx[i]
        k = len(indices[indptr[i]: indptr[i+1]]) - 1
        neigh_feature = np.zeros(shape=(features.shape[1]))
        for neighbor_idx in indices[indptr[i]: indptr[i+1]]:
            # if neighbor_idx != central_node:
            #     neigh_feature += 1/k * ((1-gama) * features[neighbor_idx] + gama * features[central_node])
            if neighbor_idx != central_node:
                for j in prange(features.shape[1]):
                    neigh_feature[j] += 1 / k * ((1-gama) * features[neighbor_idx][j] + gama * features[central_node][j])
        all_map[i] = neigh_feature
    return all_map


