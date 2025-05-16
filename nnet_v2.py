import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score


class Edge_Sheaf_NNet(nn.Module):
    def __init__(self, nvert, dimx, nlab, nconv=3, nsmat=64):
        super(Edge_Sheaf_NNet, self).__init__()
        self.nvert = nvert
        self.nconv = nconv
        self.dimx = dimx
        self.nlab = nlab
        self.cl_smat = nn.Sequential(nn.Linear(self.dimx, self.nlab),
                                     nn.LogSoftmax(dim=1))

        self.fc_smat = nn.Sequential(nn.Linear(self.dimx * 2, nsmat),
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
        print(self.fc_smat)

    def get_edge_matrix(self, source_features, target_features):
        concat_features = torch.cat([source_features, target_features], dim=1)
        edge_matrix_flat = self.fc_smat(concat_features)
        edge_matrix = torch.reshape(edge_matrix_flat, (-1, self.dimx, self.dimx))
        return edge_matrix

    def forward(self, xembed, ylabel, ylprob, wgraph, idvert):
        loss = 0.0
        xmaped = 0.0 + xembed
        num_vertices = wgraph.shape[0]
        edge_indices = []
        reverse_edge_pairs = []
        
        for i in range(num_vertices):
            for j in range(num_vertices):
                if wgraph[i, j] > 0:
                    edge_indices.append((i, j))
                    if wgraph[j, i] > 0 and i < j:
                        reverse_edge_pairs.append((i, j, j, i))
        
        num_edges = len(edge_indices)
        
        if num_edges > 0:
            source_indices = [i for i, j in edge_indices]
            target_indices = [j for i, j in edge_indices]
            
            source_features = xembed[source_indices]
            target_features = xembed[target_indices]
            
            edge_matrices_batch = self.get_edge_matrix(source_features, target_features)
            
            edge_matrices = {}
            for idx, (i, j) in enumerate(edge_indices):
                edge_matrices[(i, j)] = edge_matrices_batch[idx:idx+1]
        else:
            edge_matrices = {}
        
        loss_orth = 0.0
        if len(reverse_edge_pairs) > 0:
            for i1, j1, i2, j2 in reverse_edge_pairs:
                # AB = I check
                prod = torch.bmm(edge_matrices[(i1, j1)], edge_matrices[(i2, j2)])
                target_matrix = torch.eye(self.dimx).unsqueeze(0).to(prod.device)
                diff = prod - target_matrix
                loss_orth += torch.sqrt(torch.mean(diff * diff) * self.dimx * self.dimx)
            
            loss_orth /= len(reverse_edge_pairs)
        
        for idx_conv in range(self.nconv):
            new_xmaped = torch.zeros_like(xmaped)
            node_counts = torch.zeros(num_vertices).to(xmaped.device)
            
            if num_edges > 0:
                updated_source_features = xmaped[source_indices]
                
                messages = torch.bmm(
                    edge_matrices_batch,
                    updated_source_features.unsqueeze(2)
                ).squeeze(2)
                
                for idx, (i, j) in enumerate(edge_indices):
                    weight = wgraph[i, j]
                    new_xmaped[j:j+1, :] += weight * messages[idx:idx+1]
                    node_counts[j] += weight
            
            for j in range(num_vertices):
                if node_counts[j] > 0:
                    new_xmaped[j] /= node_counts[j]
            
            xmaped = new_xmaped
            
            if num_edges > 0:
                source_features = xmaped[source_indices]
                target_features = xmaped[target_indices]
                
                edge_matrices_batch = self.get_edge_matrix(source_features, target_features)
                
                for idx, (i, j) in enumerate(edge_indices):
                    edge_matrices[(i, j)] = edge_matrices_batch[idx:idx+1]
        
        loss_smap = torch.mean((xmaped - xembed) * (xmaped - xembed)) * self.dimx
        
        if idvert.size > 0:
            glprob = self.cl_smat(xmaped[idvert[:], :])
            
            ylprob_selected = ylprob[idvert[:], :]
            min_classes = min(ylprob_selected.shape[1], glprob.shape[1])
            
            ylprob_selected = ylprob_selected[:, :min_classes]
            glprob = glprob[:, :min_classes]
            
            kl_div = torch.sum(torch.exp(ylprob_selected) * (ylprob_selected - glprob), 1)
            
            loss_lbpr = torch.mean(kl_div)
        else:
            loss_lbpr = torch.tensor(0.0, requires_grad=True, device=xembed.device)
        
        loss_cons = 0.0
        if num_edges > 0:
            final_source_features = xmaped[source_indices]
            final_target_features = xmaped[target_indices]
            final_edge_matrices_batch = self.get_edge_matrix(final_source_features, final_target_features)
            diff = (final_edge_matrices_batch - edge_matrices_batch) ** 2
            loss_cons = torch.mean(diff) * self.dimx * self.dimx
        
        if num_edges > 0:
            loss_cons /= num_edges
        
        if idvert.size > 0:
            yscore = self.cl_smat(xmaped[idvert[:], :]).cpu().detach().numpy()
            ynumer = np.zeros((yscore.shape[0]))
            for idx in range(ynumer.size):
                max_val = 0.0
                for idx_max in range(yscore.shape[1]):
                    if max_val < np.exp(yscore[idx, idx_max]):
                        max_val = np.exp(yscore[idx, idx_max])
                        ynumer[idx] = idx_max
            loss_accs = accuracy_score(ylabel[idvert[:]], ynumer)
        else:
            loss_accs = 0.0
        
        return (loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs)

    def label_inference(self, xembed, wgraph, idx_target):
        xmaped = 0.0 + xembed
        num_vertices = wgraph.shape[0]
        
        edge_indices = []
        for i in range(num_vertices):
            for j in range(num_vertices):
                if wgraph[i, j] > 0:
                    edge_indices.append((i, j))
        
        for idx_conv in range(self.nconv):
            if len(edge_indices) > 0:
                source_indices = [i for i, j in edge_indices]
                target_indices = [j for i, j in edge_indices]
                
                source_features = xmaped[source_indices]
                target_features = xmaped[target_indices]
                
                edge_matrices_batch = self.get_edge_matrix(source_features, target_features)
                
                new_xmaped = torch.zeros_like(xmaped)
                node_counts = torch.zeros(num_vertices).to(xmaped.device)
                
                messages = torch.bmm(
                    edge_matrices_batch,
                    source_features.unsqueeze(2)
                ).squeeze(2)
                
                for idx, (i, j) in enumerate(edge_indices):
                    weight = wgraph[i, j]
                    new_xmaped[j:j+1, :] += weight * messages[idx:idx+1]
                    node_counts[j] += weight
                
                for j in range(num_vertices):
                    if node_counts[j] > 0:
                        new_xmaped[j] /= node_counts[j]
                
                xmaped = new_xmaped
            else:
                pass
        
        glprob = self.cl_smat(xmaped)
        return np.argmax(glprob[idx_target, :].cpu().detach().numpy())