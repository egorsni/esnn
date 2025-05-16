import numpy as np
import torch
from torch import nn
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score


class Edge_Sheaf_NNet(nn.Module):
    def __init__(self, nvert, dimx, dims, nlab, nconv=3, nsmat=64):
        super(Edge_Sheaf_NNet, self).__init__()
        self.nvert = nvert
        self.nconv = nconv
        self.dimx = dimx
        self.nlab = nlab
        self.dims = dims
        
        self.cl_smat = nn.Sequential(nn.Linear(self.dimx, self.nlab),
                                     nn.LogSoftmax(dim=1))

        self.fc_smat = nn.Sequential(nn.Linear(self.dims * 2, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, self.dimx * self.dimx))


    def get_edge_matrix(self, source_features, target_features):
        concat_features = torch.cat([source_features, target_features], dim=1)
        edge_matrix_flat = self.fc_smat(concat_features)
        return edge_matrix_flat.view(-1, self.dimx, self.dimx)

    def preprocess_graph(self, wgraph):
        device = wgraph.device
        num_vertices = wgraph.shape[0]
        
        indices = torch.nonzero(wgraph > 0, as_tuple=True)
        edge_indices = list(zip(indices[0].tolist(), indices[1].tolist()))
        
        source_indices = torch.tensor([i for i, j in edge_indices], device=device)
        target_indices = torch.tensor([j for i, j in edge_indices], device=device)
        edge_weights = wgraph[source_indices, target_indices]
        
        # adjacency_sparse = SparseTensor(
        #     row=target_indices,
        #     col=source_indices,
        #     value=edge_weights,
        #     sparse_sizes=(num_vertices, num_vertices)
        # )
        
        reverse_edge_pairs = []
        edge_dict = {(i, j): idx for idx, (i, j) in enumerate(edge_indices)}
        
        for (i, j), idx in edge_dict.items():
            if i < j and (j, i) in edge_dict:
                reverse_edge_pairs.append((idx, edge_dict[(j, i)]))
        
        return {
            'edge_indices': edge_indices,
            'source_indices': source_indices,
            'target_indices': target_indices,
            'edge_weights': edge_weights.to(torch.float),
            # 'adjacency_sparse': adjacency_sparse,
            'reverse_edge_pairs': reverse_edge_pairs,
            'num_edges': len(edge_indices)
        }

    def forward(self, xembed, sembed, ylabel, ylprob, wgraph, idvert):
        graph_data = self.preprocess_graph(wgraph)
        num_vertices = wgraph.shape[0]
        
        xmaped = xembed.clone()
        
        if graph_data['num_edges'] == 0:
            loss_orth = torch.tensor(0.0, device=xembed.device)
            loss_cons = torch.tensor(0.0, device=xembed.device)
            edge_matrices_batch = None
            final_edge_matrices_batch = None
        else:
            source_features = sembed[graph_data['source_indices']]
            target_features = sembed[graph_data['target_indices']]
            
            initial_edge_matrices = self.get_edge_matrix(source_features, target_features)
            
            loss_orth = torch.tensor(0.0, device=xembed.device)
            if len(graph_data['reverse_edge_pairs']) > 0:
                idx1_list, idx2_list = zip(*graph_data['reverse_edge_pairs'])
                idx1_tensor = torch.tensor(idx1_list, device=xembed.device)
                idx2_tensor = torch.tensor(idx2_list, device=xembed.device)
                
                matrices_A = initial_edge_matrices[idx1_tensor]
                matrices_B = initial_edge_matrices[idx2_tensor]
                
                prod = torch.bmm(matrices_A, matrices_B)
                target_matrix = torch.eye(self.dimx, device=prod.device).unsqueeze(0).expand(prod.shape[0], -1, -1)
                
                diff = prod - target_matrix
                loss_orth = torch.mean(torch.sqrt(torch.mean(diff * diff, dim=(1, 2)) * self.dimx * self.dimx))
        
        if graph_data['num_edges'] > 0:
            source_features = sembed[graph_data['source_indices']]
            target_features = sembed[graph_data['target_indices']]
            edge_matrices_batch = self.get_edge_matrix(source_features, target_features)
            # print(edge_matrices_batch.shape)
            # print(source_features.shape)
            # print(source_features.unsqueeze(2).shape)
            messages = torch.bmm(edge_matrices_batch, xmaped[graph_data['source_indices']].unsqueeze(2)).squeeze(2)
            
            weighted_messages = messages * graph_data['edge_weights'].unsqueeze(1)
            
            new_xmaped = torch.zeros_like(xmaped)
            new_xmaped = new_xmaped.index_add_(0, graph_data['target_indices'], weighted_messages)
            
            degrees = torch.zeros(num_vertices, device=xmaped.device)
            degrees.index_add_(0, graph_data['target_indices'], graph_data['edge_weights'])
            valid_nodes = degrees > 0
            new_xmaped[valid_nodes] = new_xmaped[valid_nodes] / degrees[valid_nodes].unsqueeze(1)
            
            xmaped = new_xmaped
        
        loss_smap = torch.mean((xmaped - xembed) ** 2) * self.dimx
        
        if idvert.size > 0:
            glprob = self.cl_smat(xmaped[idvert])
            
            ylprob_selected = ylprob[idvert]
            min_classes = min(ylprob_selected.shape[1], glprob.shape[1])
            
            ylprob_selected = ylprob_selected[:, :min_classes]
            glprob = glprob[:, :min_classes]
            
            kl_div = torch.sum(torch.exp(ylprob_selected) * (ylprob_selected - glprob), dim=1)
            loss_lbpr = torch.mean(kl_div)
            
            with torch.no_grad():
                yscore = glprob.cpu().numpy()
                ypred = np.argmax(yscore, axis=1)
                loss_accs = accuracy_score(ylabel[idvert], ypred)
        else:
            loss_lbpr = torch.tensor(0.0, requires_grad=True, device=xembed.device)
            loss_accs = 0.0
        
        loss_cons = torch.tensor(0.0, device=xembed.device)
        if graph_data['num_edges'] > 0:
            final_source_features = sembed[graph_data['source_indices']]
            final_target_features = sembed[graph_data['target_indices']]
            final_edge_matrices_batch = self.get_edge_matrix(final_source_features, final_target_features)
            
            loss_cons = torch.mean((final_edge_matrices_batch - initial_edge_matrices) ** 2) * self.dimx * self.dimx
        
        return (loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs)

    def label_inference(self, xembed, sembed, wgraph, idx_target):
        graph_data = self.preprocess_graph(wgraph)
        xmaped = xembed.clone()
        num_vertices = wgraph.shape[0]
        
        for idx_conv in range(self.nconv):
            if graph_data['num_edges'] > 0:
                source_features = sembed[graph_data['source_indices']]
                target_features = sembed[graph_data['target_indices']]
                
                edge_matrices_batch = self.get_edge_matrix(source_features, target_features)
                
                messages = torch.bmm(
                    edge_matrices_batch,
                    xmaped[graph_data['target_indices']].unsqueeze(2)
                ).squeeze(2)
                
                # new_xmaped = torch.zeros_like(xmaped)
                # node_counts = torch.zeros(wgraph.shape[0], device=xmaped.device)
                
                # for idx, j in enumerate(graph_data['target_indices']):
                #     weight = graph_data['edge_weights'][idx]
                #     new_xmaped[j] += weight * messages[idx]
                #     node_counts[j] += weight
                
                # valid_nodes = node_counts > 0
                # new_xmaped[valid_nodes] = new_xmaped[valid_nodes] / node_counts[valid_nodes].unsqueeze(1)
                
                # xmaped = new_xmaped
                weighted_messages = messages * graph_data['edge_weights'].unsqueeze(1)

                new_xmaped = torch.zeros_like(xmaped)
                new_xmaped = new_xmaped.index_add_(0, graph_data['target_indices'], weighted_messages)

                degrees = torch.zeros(num_vertices, device=xmaped.device)
                degrees.index_add_(0, graph_data['target_indices'], graph_data['edge_weights'])
                valid_nodes = degrees > 0
                new_xmaped[valid_nodes] = new_xmaped[valid_nodes] / degrees[valid_nodes].unsqueeze(1)

                xmaped = new_xmaped
        
        glprob = self.cl_smat(xmaped)
        return torch.argmax(glprob[idx_target].cpu()).item()