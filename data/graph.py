import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
            
        print("norm_adj_mat.shape")
        print(norm_adj_mat.shape)
        print("inter:",adj_mat.sum())


        n=adj_mat.shape[0]
        # m=adj.sum(1).sum()/2.0
        m=adj_mat.sum()
        row=np.array(adj_mat.sum(1)).reshape(-1)

        Ai=[ np.power(row[i],0.5) for i in range(n)]
        Aj=[ np.power(row[j],0.5) for j in range(n)]
        Ai=np.array(Ai).reshape(n,1)
        Aj=np.array(Aj).reshape(1,n)*(1.0/(2*m))
        print("Ai/Aj computed, symmetric normalization")
        # Ai,Aj=torch.tensor(Ai,dtype=torch.float32,requires_grad=False),torch.tensor(Aj,dtype=torch.float32,requires_grad=False)
        # Ai,Aj=Ai.to_sparse(),Aj.to_sparse()
        # Ai,Aj=Ai.to(self.norm_adj_mat.device),Aj.to(self.norm_adj_mat.device)
        return norm_adj_mat,Ai,Aj

    

    def convert_to_laplacian_mat(self, adj_mat):
        pass
