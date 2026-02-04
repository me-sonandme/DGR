import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import numpy as np
# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['XSimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

        ii_neg_neighbor_num=10
        ii_neighbor_num=10
        self.lambda_1=1e-1
        self.lambda_2=5e-6
        train_mat=self.data.train_mat
        self.ii_neg_neighbor_mat,self.ii_neg_constraint_mat=self.get_ii_neg_constraint_mat(train_mat,ii_neg_neighbor_num)
        self.ii_neighbor_mat, self.ii_constraint_mat =self.get_ii_constraint_mat(train_mat, ii_neighbor_num)
    
    def get_ii_neg_constraint_mat(self,train_mat,ii_neg_neighbor_num):
        A=train_mat.T.dot(train_mat)
        n_items=A.shape[0]
        res_mat=torch.zeros((n_items,ii_neg_neighbor_num))
        res_sim_mat=torch.zeros((n_items,ii_neg_neighbor_num))

        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1) 
        
        beta_uD = (np.sqrt(users_D + 1) / (users_D+1)).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        
        beta_uD[np.isinf(beta_uD)] = 0.
        beta_iD[np.isinf(beta_iD)] = 0.
        
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  #m*m
        
        for i in range(n_items):
            # Search in this row
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            
            # Get non-zero elements of this row
            non_zero_idx=torch.nonzero(row)
            idx=non_zero_idx[:,0].reshape(-1)
            #print(idx.size())
            row_nonzero=row[idx]
            
            
            idx = torch.randperm(row_nonzero.nelement())
            row_nonzero = row_nonzero[idx]
            # print(idx.size())
            
            if row_nonzero.size()[0] < 3*ii_neg_neighbor_num:
                res_mat[i]=torch.arange(ii_neg_neighbor_num)
                res_sim_mat[i]=torch.zeros(ii_neg_neighbor_num)
            else:
                row_sims, row_idxs = torch.topk(-1.0*row_nonzero, ii_neg_neighbor_num)
                # print("neg",row_sims)
                # print("neg",row_idxs)
                # print("nonzero",torch.nonzero(row_sims).size())
                res_mat[i]=idx[row_idxs]
                res_sim_mat[i]=-1.0*row_sims
            
        return res_mat.long(), res_sim_mat.float()
    
    def get_ii_constraint_mat(self,train_mat, num_neighbors, ii_diagonal_zero = False):
        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)	# Transpose n*m to m*n, result is m*m item-item co-occurrence matrix
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:  # Set diagonal to zero (self co-occurrence)
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1)  # Row sum (item degrees in co-occurrence matrix)
        
        beta_uD = (np.sqrt(users_D + 1) / (users_D+1)).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        
        beta_uD[np.isinf(beta_uD)] = 0.
        beta_iD[np.isinf(beta_iD)] = 0.
        
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  #m*m
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            # print("pos",row_sims)
            # print("pos",row_idxs)
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long(), res_sim_mat.float()
    

    
    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                
                desmoothing_loss=self.create_desmoothing_loss(user_idx,pos_idx,rec_user_emb, rec_item_emb)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss  + desmoothing_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    #deloss 
    def create_desmoothing_loss(self,user_idx,pos_idx,rec_user_emb, rec_item_emb):
        loss1 = self.lambda_1 * self.cal_loss_I(user_idx, pos_idx,rec_user_emb, rec_item_emb)
        loss2 = self.lambda_2*self.cal_loss_item_neg(user_idx, pos_idx,rec_user_emb, rec_item_emb)
        
        desmoothing_loss=loss1-loss2
        return desmoothing_loss

    def cal_loss_I(self, users, pos_items,user_embs,item_embs):
        #print(self.ii_neighbor_mat.size())
        neighbor_embeds = item_embs[self.ii_neighbor_mat[pos_items]]   # Shape: batch_size * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].cuda()   # Shape: batch_size * num_neighbors (normalized similarity scores)
        user_embeds = user_embs[users].unsqueeze(1)                   # Shape: batch_size * 1 * dim
        #print(user_embeds.device,neighbor_embeds.device,sim_scores.device)
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()  # Higher co-occurrence = higher weight in loss
        # loss=torch.mean(loss)
        loss = loss.sum()
        # print(loss.item())
        return loss
    #ddl
    def cal_loss_item_neg(self,users,pos_items,user_embs,item_embs):
 
        neighbor_embeds = item_embs[self.ii_neg_neighbor_mat[pos_items]]
        sim_scores=self.ii_neg_constraint_mat[pos_items].cuda()
        user_embeds=user_embs[users].unsqueeze(1) 
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()  # Higher co-occurrence = higher weight in loss
        # loss=torch.mean(loss)
        loss = loss.sum()
       # print(loss.item())
        return loss

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        
        self.ai=data.ai
        self.aj=data.aj
        
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        
        self.ai,self.aj=torch.tensor(self.ai,dtype=torch.float32,requires_grad=False),torch.tensor(self.aj,dtype=torch.float32,requires_grad=False)
        # print(self.ai.dtype)
        # print(self.sparse_norm_adj.dtype)
        self.ai=self.ai.to_sparse().cuda()
        self.aj=self.aj.to_sparse().cuda()
        
        

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        # for k in range(self.n_layers):
        #     ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        #     if perturbed:
        #         random_noise = torch.rand_like(ego_embeddings).cuda()
        #         ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
        #     all_embeddings.append(ego_embeddings)
        #     if k==self.layer_cl-1:
        #         all_embeddings_cl = ego_embeddings
        
        all_emb1= torch.sparse.mm(self.sparse_norm_adj,ego_embeddings)
        m=torch.sparse.mm(self.aj,all_emb1)
        m=torch.sparse.mm(self.ai,m)
        all_emb1=1.1*all_emb1-0.1*m
        random_noise = torch.rand_like(all_emb1).cuda()
        all_emb1 += torch.sign(all_emb1) * F.normalize(random_noise, dim=-1) * self.eps
        all_embeddings.append(all_emb1)
        
        all_emb2=torch.sparse.mm(self.sparse_norm_adj, all_emb1)
        m=torch.sparse.mm(self.aj,all_emb2)
        m=torch.sparse.mm(self.ai,m)
        all_emb2=2.1*all_emb2-1.1*m
        random_noise = torch.rand_like(all_emb2).cuda()
        all_emb2 += torch.sign(all_emb2) * F.normalize(random_noise, dim=-1) * self.eps
        all_embeddings.append(all_emb2)
        
        all_emb3=torch.sparse.mm(self.sparse_norm_adj, all_emb2)
        m=torch.sparse.mm(self.aj,all_emb3)
        m=torch.sparse.mm(self.ai,m)
        all_emb3=1.1*all_emb3-0.1*m
        random_noise = torch.rand_like(all_emb3).cuda()
        all_emb3 += torch.sign(all_emb3) * F.normalize(random_noise, dim=-1) * self.eps
        all_embeddings.append(all_emb3)
        
        all_emb4=torch.sparse.mm(self.sparse_norm_adj, all_emb3)
        m=torch.sparse.mm(self.aj,all_emb4)
        m=torch.sparse.mm(self.ai,m)
        all_emb4=1.05*all_emb4-0.05*m
        random_noise = torch.rand_like(all_emb4).cuda()
        all_emb4 += torch.sign(all_emb4) * F.normalize(random_noise, dim=-1) * self.eps
        all_embeddings.append(all_emb4)        
          
        all_embeddings_cl = all_emb3       
                
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
