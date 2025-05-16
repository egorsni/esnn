import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def compute_sparse_matrix_vector_product(smat, iidx, jidx, u):
    v = np.zeros((u.size))
    for idx in range(smat.size):
        v[iidx[idx]] += smat[idx] * u[jidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_sparse_transpose_matrix_vector_product(smat, iidx, jidx, u):
    v = np.zeros((u.size))
    for idx in range(smat.size):
        v[jidx[idx]] += smat[idx] * u[iidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, u):
    numu, dimu = np.shape(u)
    dimv = np.max(iidx) + 1
    v = np.zeros((numu, dimv))
    for idx in range(smat.size):
        v[:, iidx[idx]] += smat[idx] * u[:, jidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_sparse_transpose_matrix_dense_matrix_product(smat, iidx, jidx, u):
    numu, dimu = np.shape(u)
    dimv = np.max(jidx) + 1
    v = np.zeros((numu, dimv))
    for idx in range(smat.size):
        v[:, jidx[idx]] += smat[idx] * u[:, iidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def perform_orthogonalization(ymat, eps=1.0e-6):
    nvec, dimy = ymat.shape
    umat = ymat.copy()
    for idx in range(nvec):
        umat[idx, :] = umat[idx, :] / max(eps, np.sqrt(np.sum(umat[idx, :] * umat[idx, :])))
        for idx_orth in range(idx + 1, nvec):
            product_value = np.dot(umat[idx, :], umat[idx_orth, :])
            umat[idx_orth, :] = umat[idx_orth, :] - product_value * umat[idx, :]
    return umat

@jit(nopython=True, fastmath=True)
def compute_qr_factorization(smat, iidx, jidx, nvec, dimy, niter, eps=1.0e-6):
    rmat = np.random.normal(0.0, 1.0, (nvec, dimy))
    rmat = perform_orthogonalization(rmat)

    for kiter in range(niter):
        qmat = compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, rmat)
        umat = compute_sparse_transpose_matrix_dense_matrix_product(smat, iidx, jidx, qmat)
        zmat = perform_orthogonalization(umat)
        diff = np.max(np.absolute(zmat - rmat))
        rmat = 0.0 + zmat
        cmat = np.dot(rmat, np.transpose(rmat))
        if diff < eps:
            break
    qmat = compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, rmat)
    qmat = perform_orthogonalization(qmat)
    return (qmat, rmat)


def convert_edge_index_to_list(edge_index):
    sorted_indices = np.argsort(edge_index[0])
    sorted_edge_index = edge_index[:, sorted_indices]
    
    num_nodes = int(edge_index.max()) + 1
    edge_list = [[] for _ in range(num_nodes)]
    
    nedge = sorted_edge_index.shape[1]
    current_src = -1
    
    for idx_edge in range(nedge):
        src = sorted_edge_index[0, idx_edge]
        dst = sorted_edge_index[1, idx_edge]
        
        if src != current_src:
            current_src = src
        
        edge_list[src].append(dst)
    
    return edge_list



# def graph_random_walk(edge_list, nsample):
#     random_walk_data = np.zeros((nsample))
#     nvert = len(edge_list)
#     idx_vert = np.random.randint(0, high=nvert)

#     while len(edge_list[idx_vert]) == 0:
#         idx_vert = np.random.randint(0, high=nvert)
    
#     for idx in range(nsample):
#         random_walk_data[idx] = idx_vert
#         vert_list = edge_list[idx_vert]
#         idx_vert = vert_list[np.random.randint(0, high=len(vert_list))]
#     return random_walk_data

def graph_random_walk(edge_list, nsample):
    random_walk_data = np.zeros((nsample))
    nvert = len(edge_list)
    
    non_empty_vertices = [i for i in range(nvert) if len(edge_list[i]) > 0]
    
    if len(non_empty_vertices) == 0:
        random_walk_data = np.random.randint(0, nvert, size=nsample)
        return random_walk_data
    
    idx_vert = non_empty_vertices[np.random.randint(0, len(non_empty_vertices))]
    
    for idx in range(nsample):
        random_walk_data[idx] = idx_vert
        vert_list = edge_list[idx_vert]
        
        if len(vert_list) == 0:
            idx_vert = non_empty_vertices[np.random.randint(0, len(non_empty_vertices))]
        else:
            idx_vert = vert_list[np.random.randint(0, len(vert_list))]
    
    return random_walk_data


def graph_random_walk_fixed_start(edge_list, nsample, idx_start):
    random_walk_data = np.zeros((nsample))
    nvert = len(edge_list)
    idx_vert = idx_start

    while len(edge_list[idx_vert]) == 0:
        idx_vert = np.random.randint(0, high=nvert)
    
    for idx in range(nsample):
        random_walk_data[idx] = idx_vert
        vert_list = edge_list[idx_vert]
        idx_vert = vert_list[np.random.randint(0, high=len(vert_list))]
    return random_walk_data



def convert_sequence_to_graph(node_data):
    nsample = node_data.size
    node_uniq = np.unique(node_data)
    nvert = node_uniq.size
    wgraph = np.zeros((nvert, nvert))
    for isample in range(nsample - 1):
        idx0 = np.where(node_uniq == node_data[isample + 0])[0]
        idx1 = np.where(node_uniq == node_data[isample + 1])[0]
        wgraph[idx0, idx1] = 1.0
        wgraph[idx1, idx0] = 1.0
    for idx in range(wgraph.shape[0]):
            wgraph[idx, :] = wgraph[idx, :] / max(1.0, np.sum(wgraph[idx, :]))
    return (wgraph, node_uniq.astype(np.int64))




def compute_index_subsample(idx_sample, idx_target):
    set_sample = set(idx_sample.tolist())
    set_target = set(idx_target.tolist())
    inter_list = list(set_sample.intersection(set_target))
    idx_output = np.asarray(inter_list).astype(np.int64)
    # idx_output = np.zeros((len(inter_list)), dtype=np.int64)
    if len(inter_list) > 0:
        for a in range(idx_output.size):
            idx_output[a] = np.where(idx_sample == idx_output[a])[0]
    return idx_output.astype(np.int64)





